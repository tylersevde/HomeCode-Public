#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

[[nodiscard]] std::string versionString(std::uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "."
         + std::to_string(VK_API_VERSION_MINOR(version)) + "."
         + std::to_string(VK_API_VERSION_PATCH(version));
}

[[nodiscard]] bool hasExtension(
    const std::vector<VkExtensionProperties>& extensions,
    std::string_view wanted) {
    return std::any_of(extensions.begin(), extensions.end(),
        [wanted](const VkExtensionProperties& extension) {
            return wanted == extension.extensionName;
        });
}

[[nodiscard]] const char* deviceTypeName(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
    default: return "other";
    }
}

} // namespace

int main() {
    std::uint32_t loaderVersion = VK_API_VERSION_1_0;
    (void)vkEnumerateInstanceVersion(&loaderVersion);
    std::cout << "loader=" << versionString(loaderVersion) << '\n';

    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "Newton's Echo Chamber Vulkan probe";
    application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.pEngineName = "NEC";
    application.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.apiVersion = std::min(loaderVersion, VK_API_VERSION_1_3);

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &application;
    VkInstance instance{};
    const VkResult createResult = vkCreateInstance(&createInfo, nullptr, &instance);
    if (createResult != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed: " << createResult << '\n';
        return 1;
    }

    std::uint32_t deviceCount = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (result != VK_SUCCESS || deviceCount == 0) {
        std::cerr << "no Vulkan physical device: " << result << '\n';
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    (void)vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (std::uint32_t deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
        const VkPhysicalDevice device = devices[deviceIndex];
        VkPhysicalDeviceDriverProperties driver{};
        driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &driver;
        vkGetPhysicalDeviceProperties2(device, &properties2);
        const VkPhysicalDeviceProperties& properties = properties2.properties;

        std::uint32_t extensionCount = 0;
        (void)vkEnumerateDeviceExtensionProperties(
            device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        (void)vkEnumerateDeviceExtensionProperties(
            device, nullptr, &extensionCount, extensions.data());

        VkPhysicalDeviceDynamicRenderingFeatures dynamicRendering{};
        dynamicRendering.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
        VkPhysicalDeviceSynchronization2Features synchronization2{};
        synchronization2.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
        VkPhysicalDeviceMaintenance4Features maintenance4{};
        maintenance4.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        VkPhysicalDeviceVulkan12Features vulkan12{};
        vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        dynamicRendering.pNext = &synchronization2;
        synchronization2.pNext = &maintenance4;
        maintenance4.pNext = &vulkan12;
        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &dynamicRendering;
        vkGetPhysicalDeviceFeatures2(device, &features);

        std::uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, queues.data());

        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(device, &memory);

        std::cout << "device[" << deviceIndex << "]=" << properties.deviceName
                  << " type=\"" << deviceTypeName(properties.deviceType) << "\""
                  << " api=" << versionString(properties.apiVersion)
                  << " driver=" << versionString(properties.driverVersion)
                  << " driverName=\"" << driver.driverName << "\"\n";
        std::cout << "  extensions: swapchain="
                  << hasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)
                  << " dynamic_rendering="
                  << hasExtension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)
                  << " synchronization2="
                  << hasExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
                  << " maintenance4="
                  << hasExtension(extensions, VK_KHR_MAINTENANCE_4_EXTENSION_NAME)
                  << " memory_budget="
                  << hasExtension(extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)
                  << '\n';
        std::cout << "  features: dynamicRendering=" << dynamicRendering.dynamicRendering
                  << " synchronization2=" << synchronization2.synchronization2
                  << " maintenance4=" << maintenance4.maintenance4
                  << " timelineSemaphore=" << vulkan12.timelineSemaphore
                  << " shaderFloat16=" << vulkan12.shaderFloat16
                  << " bufferDeviceAddress=" << vulkan12.bufferDeviceAddress
                  << '\n';
        for (std::uint32_t queue = 0; queue < queueCount; ++queue) {
            const VkQueueFlags flags = queues[queue].queueFlags;
            std::cout << "  queue[" << queue << "] count=" << queues[queue].queueCount
                      << " graphics=" << static_cast<bool>(flags & VK_QUEUE_GRAPHICS_BIT)
                      << " compute=" << static_cast<bool>(flags & VK_QUEUE_COMPUTE_BIT)
                      << " transfer=" << static_cast<bool>(flags & VK_QUEUE_TRANSFER_BIT)
                      << '\n';
        }
        for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap) {
            const double gib = static_cast<double>(memory.memoryHeaps[heap].size)
                             / (1024.0 * 1024.0 * 1024.0);
            std::cout << "  heap[" << heap << "]=" << std::fixed
                      << std::setprecision(2) << gib << " GiB device_local="
                      << static_cast<bool>(memory.memoryHeaps[heap].flags
                                           & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                      << '\n';
        }
    }

    vkDestroyInstance(instance, nullptr);
    return 0;
}
