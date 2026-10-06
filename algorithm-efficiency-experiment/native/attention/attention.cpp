#include <vulkan/vulkan.h>
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static void check(VkResult r,const char* operation){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(operation)+": "+std::to_string(r));}
struct Shape{uint32_t n,d,b,first,rows;};
struct Buffer{VkBuffer handle=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;void* mapped=nullptr;VkDeviceSize bytes=0;};
struct Metrics{double staging_ms=0,gpu_ms=0,wait_ms=0,command_ms=0;uint64_t submissions=0,bytes=0,validation_errors=0;};
// Separate versioned ABI: the original Metrics layout and entry points stay intact.
struct ProfileMetrics{uint32_t version=1,size=sizeof(ProfileMetrics);double stages_ms[4]{},projection_ms=0,input_copy_ms=0,output_copy_ms=0,submit_ms=0,fence_ms=0,query_ms=0;};

static void projection(const float* x,const float* w,float* q,float* k,float* v,int d){
    std::fill(q,q+d,0);std::fill(k,k+d,0);std::fill(v,v+d,0);
    // Contiguous output columns permit NEON vectorization and reuse each input scalar.
    for(int z=0;z<d;z++){
        float value=x[z];
        #pragma omp simd
        for(int j=0;j<d;j++) {q[j]+=value*w[z*d+j];k[j]+=value*w[d*d+z*d+j];v[j]+=value*w[2*d*d+z*d+j];}
    }
}
static void attention_row(const float* q,const float* k,const float* v,float* out,float* score,int prefix,int d){
    float maximum=-INFINITY,scale=1.0f/std::sqrt(float(d));
    for(int j=0;j<prefix;j++){
        float sum=0;
        #pragma omp simd reduction(+:sum)
        for(int z=0;z<d;z++)sum+=q[z]*k[j*d+z];
        score[j]=sum*scale;maximum=std::max(maximum,score[j]);
    }
    float denominator=0;
    for(int j=0;j<prefix;j++){score[j]=std::exp(score[j]-maximum);denominator+=score[j];}
    std::fill(out,out+d,0);
    for(int j=0;j<prefix;j++){
        float a=score[j]/denominator;
        #pragma omp simd
        for(int z=0;z<d;z++)out[z]+=a*v[j*d+z];
    }
}

struct Engine{
    VkInstance instance=VK_NULL_HANDLE;VkPhysicalDevice physical=VK_NULL_HANDLE;VkDevice device=VK_NULL_HANDLE;
    VkQueue queue=VK_NULL_HANDLE;VkCommandPool pool=VK_NULL_HANDLE;VkCommandBuffer command=VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool=VK_NULL_HANDLE;VkDescriptorSet descriptors=VK_NULL_HANDLE;
    VkQueryPool queries=VK_NULL_HANDLE;VkFence fence=VK_NULL_HANDLE;VkDebugUtilsMessengerEXT debug=VK_NULL_HANDLE;
    VkPipeline pipelines[5][4][2]{};Buffer buffers[7];VkPhysicalDeviceProperties properties{};
    ProfileMetrics profile;bool profiling=false;int kernel_variant=0;
    std::atomic<uint64_t> validation_errors{0};Metrics metrics;std::string error,identity;
    std::vector<float> cq,ck,cv,cs;int n=0,d=0,b=0;bool gpu=false;
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,const VkDebugUtilsMessengerCallbackDataEXT* data,void* user){
        if(severity&VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)static_cast<Engine*>(user)->validation_errors++;
        std::cerr<<"Vulkan validation: "<<data->pMessage<<std::endl;return VK_FALSE;
    }
    void init(const std::string& shader_dir,bool validation){
        gpu=true;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="attention-feedback";app.apiVersion=VK_API_VERSION_1_2;
        const char* layer="VK_LAYER_KHRONOS_validation";
        const char* extensions[]={VK_EXT_DEBUG_UTILS_EXTENSION_NAME,VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME};
        VkValidationFeatureEnableEXT enabled=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
        VkValidationFeaturesEXT features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};features.enabledValidationFeatureCount=1;features.pEnabledValidationFeatures=&enabled;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};info.pApplicationInfo=&app;
        if(validation){info.enabledLayerCount=1;info.ppEnabledLayerNames=&layer;info.enabledExtensionCount=2;info.ppEnabledExtensionNames=extensions;info.pNext=&features;}
        check(vkCreateInstance(&info,nullptr,&instance),"create instance");
        if(validation){
            VkDebugUtilsMessengerCreateInfoEXT di{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            di.messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            di.messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            di.pfnUserCallback=debug_message;di.pUserData=this;
            auto create=(PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,"vkCreateDebugUtilsMessengerEXT");
            if(!create)throw std::runtime_error("Missing Vulkan debug callback");check(create(instance,&di,nullptr,&debug),"debug messenger");
        }
        uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr),"enumerate count");
        std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance,&count,devices.data()),"enumerate devices");
        for(auto candidate:devices){VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(candidate,&p);
            if(p.vendorID==0x14e4 && p.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && std::string(p.deviceName).find("V3D")==0){physical=candidate;properties=p;break;}}
        if(!physical)throw std::runtime_error("Hardware V3D GPU required; software fallback is forbidden");
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());uint32_t family=count;
        for(uint32_t i=0;i<count;i++)if((families[i].queueFlags&VK_QUEUE_COMPUTE_BIT)&&families[i].timestampValidBits==64){family=i;break;}
        if(family==count || properties.limits.maxComputeWorkGroupInvocations<256)throw std::runtime_error("Compute/timestamp limits unsupported");
        identity=std::string(properties.deviceName)+"; vendor="+std::to_string(properties.vendorID)+"; driver="+std::to_string(properties.driverVersion);
        float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dc.queueCreateInfoCount=1;dc.pQueueCreateInfos=&qi;
        check(vkCreateDevice(physical,&dc,nullptr,&device),"create device");vkGetDeviceQueue(device,family,0,&queue);
        VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pc.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;pc.queueFamilyIndex=family;
        check(vkCreateCommandPool(device,&pc,nullptr,&pool),"command pool");
        VkCommandBufferAllocateInfo ac{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ac.commandPool=pool;ac.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ac.commandBufferCount=1;
        check(vkAllocateCommandBuffers(device,&ac,&command),"command buffer");
        VkDescriptorSetLayoutBinding bindings[7]{};
        for(int i=0;i<7;i++){bindings[i].binding=i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
        VkDescriptorSetLayoutCreateInfo sc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sc.bindingCount=7;sc.pBindings=bindings;
        check(vkCreateDescriptorSetLayout(device,&sc,nullptr,&set_layout),"set layout");
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(Shape)};
        VkPipelineLayoutCreateInfo lc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};lc.setLayoutCount=1;lc.pSetLayouts=&set_layout;lc.pushConstantRangeCount=1;lc.pPushConstantRanges=&range;
        check(vkCreatePipelineLayout(device,&lc,nullptr,&layout),"pipeline layout");
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,7};VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=1;dp.poolSizeCount=1;dp.pPoolSizes=&size;
        check(vkCreateDescriptorPool(device,&dp,nullptr,&descriptor_pool),"descriptor pool");
        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};da.descriptorPool=descriptor_pool;da.descriptorSetCount=1;da.pSetLayouts=&set_layout;
        check(vkAllocateDescriptorSets(device,&da,&descriptors),"descriptor set");
        VkQueryPoolCreateInfo qc{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qc.queryType=VK_QUERY_TYPE_TIMESTAMP;qc.queryCount=10;
        check(vkCreateQueryPool(device,&qc,nullptr,&queries),"query pool");
        VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(device,&fc,nullptr,&fence),"fence");
        const char* names[]={"project","scores","softmax","apply"};
        for(int kernel=0;kernel<5;kernel++)for(int stage=0;stage<4;stage++)for(int variant=0;variant<2;variant++){
            std::string suffix=(kernel<3 && kernel && stage)?"-v"+std::to_string(kernel):"";
            std::string path=shader_dir+"/"+names[stage]+std::to_string(stage==2?(variant?128:64):(variant?16:8))+suffix+".spv";
            if(kernel>=3 && stage!=2)path=shader_dir+"/"+names[stage]+"-stream"+std::to_string(kernel==3?32:64)+".spv";
            std::ifstream stream(path,std::ios::binary|std::ios::ate);if(!stream)throw std::runtime_error("Missing shader "+path);
            size_t bytes=stream.tellg();if(!bytes || bytes%4)throw std::runtime_error("Invalid SPIR-V size");
            std::vector<uint32_t> code(bytes/4);stream.seekg(0);stream.read((char*)code.data(),bytes);if(!stream)throw std::runtime_error("Shader read failed");
            VkShaderModuleCreateInfo mc{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mc.codeSize=bytes;mc.pCode=code.data();VkShaderModule module;
            check(vkCreateShaderModule(device,&mc,nullptr,&module),"shader module");
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.layout=layout;
            ci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";
            VkResult result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&ci,nullptr,&pipelines[kernel][stage][variant]);vkDestroyShaderModule(device,module,nullptr);check(result,"compute pipeline");
        }
    }
    void free_buffers(){for(auto& x:buffers){if(x.mapped)vkUnmapMemory(device,x.memory);if(x.handle)vkDestroyBuffer(device,x.handle,nullptr);if(x.memory)vkFreeMemory(device,x.memory,nullptr);x=Buffer{};}}
    void configure(int nn,int dd,int bb){
        if(nn<1||nn>4096||dd<1||dd>128||bb<1||bb>16)throw std::runtime_error("Shape outside declared bounds");
        if((size_t(5)*nn*dd*bb+size_t(3)*dd*dd+size_t(nn)*nn*bb)*sizeof(float)>512ull*1024*1024)
            throw std::runtime_error("Shape exceeds 512 MiB workspace budget");
        if(n==nn&&d==dd&&b==bb)return;
        n=nn;d=dd;b=bb;size_t elements=size_t(n)*d*b;
        cq.resize(elements);ck.resize(elements);cv.resize(elements);cs.resize(size_t(n)*n*b);
        if(!gpu)return;
        free_buffers();size_t sizes[]={elements,size_t(3)*d*d,elements,elements,elements,size_t(n)*n*b,elements};
        VkPhysicalDeviceMemoryProperties memory;vkGetPhysicalDeviceMemoryProperties(physical,&memory);uint64_t total=0;
        for(int i=0;i<7;i++){
            auto& x=buffers[i];x.bytes=sizes[i]*sizeof(float);
            VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bc.size=x.bytes;bc.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
            check(vkCreateBuffer(device,&bc,nullptr,&x.handle),"create buffer");VkMemoryRequirements requirements;vkGetBufferMemoryRequirements(device,x.handle,&requirements);
            total+=requirements.size;if(total>512ull*1024*1024)throw std::runtime_error("GPU buffers exceed 512 MiB budget");
            uint32_t type=memory.memoryTypeCount;
            for(uint32_t t=0;t<memory.memoryTypeCount;t++)if((requirements.memoryTypeBits&(1u<<t)) &&
                (memory.memoryTypes[t].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=t;break;}
            if(type==memory.memoryTypeCount)throw std::runtime_error("Host-coherent mapped memory unavailable");
            VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ma.allocationSize=requirements.size;ma.memoryTypeIndex=type;
            check(vkAllocateMemory(device,&ma,nullptr,&x.memory),"allocate memory");check(vkBindBufferMemory(device,x.handle,x.memory,0),"bind buffer");
            check(vkMapMemory(device,x.memory,0,x.bytes,0,&x.mapped),"map memory");
        }
        metrics.bytes=total;VkDescriptorBufferInfo info[7]{};VkWriteDescriptorSet writes[7]{};
        for(int i=0;i<7;i++){info[i]={buffers[i].handle,0,buffers[i].bytes};writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet=descriptors;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&info[i];}
        vkUpdateDescriptorSets(device,7,writes,0,nullptr);
    }
    void cpu(const float* x,const float* w,float* out,bool streaming,int threads){
        if(threads!=1&&threads!=4)throw std::runtime_error("CPU threads must be one or four");
        if(!streaming){
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(int row=0;row<b*n;row++)projection(x+row*d,w,cq.data()+row*d,ck.data()+row*d,cv.data()+row*d,d);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(int row=0;row<b*n;row++)attention_row(cq.data()+row*d,ck.data()+(row/n)*n*d,cv.data()+(row/n)*n*d,out+row*d,cs.data()+row*n,row%n+1,d);
        }else{
            for(int t=0;t<n;t++){
                #pragma omp parallel for num_threads(threads) schedule(static)
                for(int batch=0;batch<b;batch++){
                    int row=batch*n+t;projection(x+row*d,w,cq.data()+row*d,ck.data()+row*d,cv.data()+row*d,d);
                    attention_row(cq.data()+row*d,ck.data()+batch*n*d,cv.data()+batch*n*d,out+row*d,cs.data()+row*n,t+1,d);
                }
                // All streams complete this step before the next vector is accessed.
            }
        }
    }
    void barrier(VkAccessFlags src,VkAccessFlags dst,VkPipelineStageFlags from,VkPipelineStageFlags to){
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};mb.srcAccessMask=src;mb.dstAccessMask=dst;
        vkCmdPipelineBarrier(command,from,to,0,1,&mb,0,nullptr,0,nullptr);
    }
    void dispatch(int stage,int variant,uint32_t x,uint32_t y,uint32_t z){
        if(profiling)vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,2+stage*2);
        vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipelines[kernel_variant][stage][variant]);vkCmdDispatch(command,x,y,z);
        barrier(VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        if(profiling)vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,3+stage*2);
    }
    void compute(const Shape& shape,int candidate){
        auto start=Clock::now();check(vkResetCommandBuffer(command,0),"reset command");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command,&bi),"begin command");
        vkCmdResetQueryPool(command,queries,0,profiling?10:2);
        barrier(VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,queries,0);
        vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&descriptors,0,nullptr);
        vkCmdPushConstants(command,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(shape),&shape);
        int tile_variant=(candidate>>2)&1,reduction=(candidate>>1)&1;uint32_t tile=tile_variant?16:8,limit=shape.first+shape.rows;
        if(kernel_variant>=3){
            if(shape.rows!=1)throw std::runtime_error("Streaming kernels require one query");
            uint32_t lanes=kernel_variant==3?32:64;
            if(!(candidate&1))dispatch(0,tile_variant,(d+lanes-1)/lanes,1,b*3);
            dispatch(1,tile_variant,(limit+lanes-1)/lanes,1,b);
            dispatch(2,reduction,1,b,1);
            dispatch(3,tile_variant,(d+lanes-1)/lanes,1,b);
        }else{
        if(!(candidate&1))dispatch(0,tile_variant,(d+tile-1)/tile,(shape.rows+tile-1)/tile,b*3);
        dispatch(1,tile_variant,(limit+tile-1)/tile,(shape.rows+tile-1)/tile,b);
        dispatch(2,reduction,shape.rows,b,1);
        dispatch(3,tile_variant,(d+tile-1)/tile,(shape.rows+tile-1)/tile,b);
        }
        barrier(VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT);
        vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,queries,1);check(vkEndCommandBuffer(command),"end command");metrics.command_ms+=ms(start);
        start=Clock::now();check(vkResetFences(device,1,&fence),"reset fence");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&command;
        check(vkQueueSubmit(queue,1,&si,fence),"submit");profile.submit_ms+=ms(start);auto waiting=Clock::now();
        check(vkWaitForFences(device,1,&fence,VK_TRUE,5000000000ull),"five-second GPU fence");profile.fence_ms+=ms(waiting);
        metrics.wait_ms+=ms(start);metrics.submissions++;
        auto query_start=Clock::now();uint64_t ticks[2];check(vkGetQueryPoolResults(device,queries,0,2,sizeof(ticks),ticks,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT),"timestamps");
        metrics.gpu_ms+=double(ticks[1]-ticks[0])*properties.limits.timestampPeriod/1e6;
        if(profiling)for(int stage=(candidate&1)?1:0;stage<4;stage++){
            uint64_t pair[2];check(vkGetQueryPoolResults(device,queries,2+stage*2,2,sizeof(pair),pair,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT),"stage timestamps");
            profile.stages_ms[stage]+=double(pair[1]-pair[0])*properties.limits.timestampPeriod/1e6;
        }
        profile.query_ms+=ms(query_start);
    }
    void run_gpu(const float* x,const float* w,float* out,bool streaming,int candidate){
        if(!gpu||candidate<0||candidate>7)throw std::runtime_error("Invalid GPU candidate");
        uint64_t bytes=metrics.bytes;metrics=Metrics{};metrics.bytes=bytes;profile=ProfileMetrics{};
        auto start=Clock::now();std::memcpy(buffers[1].mapped,w,size_t(3)*d*d*sizeof(float));metrics.staging_ms+=ms(start);profile.input_copy_ms+=ms(start);
        for(int t=0;t<(streaming?n:1);t++){
            start=Clock::now();int first=streaming?t:0,rows=streaming?1:n;
            for(int batch=0;batch<b;batch++){
                size_t offset=size_t(batch*n+first)*d;
                if(candidate&1){
                    for(int row=0;row<rows;row++)projection(x+offset+row*d,w,
                        (float*)buffers[2].mapped+offset+row*d,(float*)buffers[3].mapped+offset+row*d,(float*)buffers[4].mapped+offset+row*d,d);
                }else std::memcpy((float*)buffers[0].mapped+offset,x+offset,size_t(rows)*d*sizeof(float));
            }
            double staged=ms(start);metrics.staging_ms+=staged;if(candidate&1)profile.projection_ms+=staged;else profile.input_copy_ms+=staged;
            compute(Shape{uint32_t(n),uint32_t(d),uint32_t(b),uint32_t(first),uint32_t(rows)},candidate);
            start=Clock::now();
            for(int batch=0;batch<b;batch++){size_t offset=size_t(batch*n+first)*d;std::memcpy(out+offset,(float*)buffers[6].mapped+offset,size_t(rows)*d*sizeof(float));}
            double copied=ms(start);metrics.staging_ms+=copied;profile.output_copy_ms+=copied;
        }
        metrics.validation_errors=validation_errors.load();
        if(metrics.validation_errors)throw std::runtime_error("Vulkan validation error; inspect native stderr");
    }
    ~Engine(){
        if(device){vkDeviceWaitIdle(device);free_buffers();for(auto& kernel:pipelines)for(auto& stage:kernel)for(auto p:stage)if(p)vkDestroyPipeline(device,p,nullptr);
            if(fence)vkDestroyFence(device,fence,nullptr);if(queries)vkDestroyQueryPool(device,queries,nullptr);
            if(descriptor_pool)vkDestroyDescriptorPool(device,descriptor_pool,nullptr);if(layout)vkDestroyPipelineLayout(device,layout,nullptr);
            if(set_layout)vkDestroyDescriptorSetLayout(device,set_layout,nullptr);if(pool)vkDestroyCommandPool(device,pool,nullptr);vkDestroyDevice(device,nullptr);}
        if(debug){auto destroy=(PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,"vkDestroyDebugUtilsMessengerEXT");if(destroy)destroy(instance,debug,nullptr);}
        if(instance)vkDestroyInstance(instance,nullptr);
    }
};

extern "C" {
void* fb_create(const char* shaders,int gpu,int validation,char* error,size_t capacity){
    try{auto e=std::make_unique<Engine>();if(gpu)e->init(shaders,validation);return e.release();}
    catch(const std::exception& ex){if(capacity){std::strncpy(error,ex.what(),capacity-1);error[capacity-1]=0;}return nullptr;}
}
int fb_configure(void* p,int n,int d,int b){auto e=(Engine*)p;try{e->configure(n,d,b);return 0;}catch(const std::exception& ex){e->error=ex.what();return -1;}}
int fb_run(void* p,const float* x,const float* w,float* out,int streaming,int backend){auto e=(Engine*)p;
    try{if(!e->n)throw std::runtime_error("Configure shape before inference");
        if(e->kernel_variant>=3 && (!streaming || backend<0))throw std::runtime_error("Streaming GPU variant requires GPU streaming");
        if(backend<0){e->metrics=Metrics{};e->profile=ProfileMetrics{};e->cpu(x,w,out,streaming,backend==-1?1:4);}else e->run_gpu(x,w,out,streaming,backend);return 0;
    }catch(const std::exception& ex){e->error=ex.what();return -1;}}
const char* fb_error(void* p){return ((Engine*)p)->error.c_str();}
const char* fb_identity(void* p){return ((Engine*)p)->identity.c_str();}
void fb_metrics(void* p,Metrics* out){*out=((Engine*)p)->metrics;}
int fb_set_options(void* p,int profiling,int variant){if((profiling!=0&&profiling!=1)||variant<0||variant>4)return -1;auto e=(Engine*)p;e->profiling=profiling;e->kernel_variant=variant;return 0;}
int fb_profile_metrics(void* p,ProfileMetrics* out,size_t capacity){if(capacity!=sizeof(ProfileMetrics))return -1;*out=((Engine*)p)->profile;return 0;}
void fb_destroy(void* p){delete (Engine*)p;}
}
