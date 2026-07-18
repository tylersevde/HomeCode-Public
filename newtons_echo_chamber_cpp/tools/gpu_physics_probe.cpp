#include "physics.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef NEC_SHADER_BINARY_DIR
#define NEC_SHADER_BINARY_DIR "."
#endif
#ifndef NEC_INSTALL_SHADER_DIR
#define NEC_INSTALL_SHADER_DIR "."
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kLocalSize = 128U;
constexpr std::uint32_t kDefaultPrepareSteps = 8U;
constexpr std::uint32_t kDefaultSamples = 12U;
constexpr std::uint32_t kDefaultWarmups = 3U;
constexpr std::uint32_t kDefaultPairCapacity = 1U << 20U;
constexpr float kDefaultCellSize = 0.25F;
constexpr float kRadiusGuard = 1.0e-4F;

struct Options {
    std::uint32_t samples{kDefaultSamples};
    std::uint32_t warmups{kDefaultWarmups};
    std::uint32_t prepareSteps{kDefaultPrepareSteps};
    std::uint32_t pairCapacity{kDefaultPairCapacity};
    float cellSize{kDefaultCellSize};
    bool allowSoftware{};
    bool mixedSleepCheck{};
    bool help{};
};

struct alignas(16) InputBody {
    std::array<float, 4> positionRadius{};
    std::array<float, 4> previousActive{};
    std::array<float, 4> velocityPad{};
};
static_assert(sizeof(InputBody) == 48U);

struct alignas(16) BroadphaseBody {
    std::array<float, 4> centerRadius{};
    std::array<std::int32_t, 4> cellCompact{};
};
static_assert(sizeof(BroadphaseBody) == 32U);

struct GpuPair {
    std::uint32_t first{};
    std::uint32_t second{};

    auto operator<=>(const GpuPair&) const = default;
};
static_assert(sizeof(GpuPair) == 8U);

struct IntegratePush {
    std::uint32_t count{};
    float deltaTime{};
    float gravity{};
    float cellSize{};
    std::uint32_t bucketMask{};
};
static_assert(sizeof(IntegratePush) == 20U);

struct BroadphasePush {
    std::uint32_t count{};
    std::uint32_t bucketMask{};
    std::uint32_t pairCapacity{};
};
static_assert(sizeof(BroadphasePush) == 12U);

struct CpuEnvelope {
    std::array<float, 3> start{};
    std::array<float, 3> end{};
    std::array<float, 3> center{};
    float baseRadius{};
    float radius{};
    bool active{};
    bool compact{};
};

struct CpuReference {
    std::vector<GpuPair> envelopePairs;
    std::size_t trueSweptPairs{};
    std::size_t conservativeExtras{};
    std::size_t trueSubsetMissing{};
    std::uint32_t compactBodies{};
    std::uint32_t largeBodies{};
    double milliseconds{};
};

struct Buffer {
    VkBuffer handle{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize size{};
    VkDeviceSize allocationSize{};
    void* mapped{};
    bool coherent{};
    bool deviceLocal{};
};

struct DeviceCandidate {
    VkPhysicalDevice device{VK_NULL_HANDLE};
    VkPhysicalDeviceProperties properties{};
    std::uint32_t queueFamily{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t timestampValidBits{};
    bool core13{};
    bool software{};
    std::uint64_t score{};
};

[[nodiscard]] std::string vkFailure(std::string_view operation, VkResult result) {
    return std::string(operation) + " failed with VkResult "
         + std::to_string(static_cast<int>(result));
}

void requireVk(VkResult result, std::string_view operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(vkFailure(operation, result));
    }
}

template <typename Structure>
[[nodiscard]] constexpr Structure vkInitialize(VkStructureType type) noexcept {
    Structure value{};
    value.sType = type;
    return value;
}

[[nodiscard]] std::string lowerCopy(std::string_view text) {
    std::string result(text);
    std::transform(result.begin(), result.end(), result.begin(), [](char value) {
        const auto byte = static_cast<unsigned char>(value);
        if (byte >= static_cast<unsigned char>('A')
            && byte <= static_cast<unsigned char>('Z')) {
            return static_cast<char>(byte - static_cast<unsigned char>('A')
                                   + static_cast<unsigned char>('a'));
        }
        return value;
    });
    return result;
}

[[nodiscard]] bool contains(std::string_view text, std::string_view value) {
    return text.find(value) != std::string_view::npos;
}

[[nodiscard]] bool isV3dv(std::string_view name) {
    const std::string lower = lowerCopy(name);
    return contains(lower, "v3d") || contains(lower, "videocore vii")
        || contains(lower, "videocore 7");
}

[[nodiscard]] bool isSoftware(const VkPhysicalDeviceProperties& properties) {
    const std::string lower = lowerCopy(properties.deviceName);
    return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU
        || contains(lower, "llvmpipe") || contains(lower, "lavapipe")
        || contains(lower, "swiftshader") || contains(lower, "software rasterizer");
}

[[nodiscard]] std::string versionString(std::uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "."
         + std::to_string(VK_API_VERSION_MINOR(version)) + "."
         + std::to_string(VK_API_VERSION_PATCH(version));
}

[[nodiscard]] bool hasExtension(
    const std::vector<VkExtensionProperties>& extensions,
    const char* wanted) {
    return std::any_of(extensions.begin(), extensions.end(), [wanted](const auto& value) {
        return std::strcmp(value.extensionName, wanted) == 0;
    });
}

[[nodiscard]] std::vector<VkExtensionProperties>
deviceExtensions(VkPhysicalDevice device) {
    std::uint32_t count = 0U;
    requireVk(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr),
              "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> values(count);
    requireVk(vkEnumerateDeviceExtensionProperties(
                  device, nullptr, &count, values.data()),
              "vkEnumerateDeviceExtensionProperties");
    values.resize(count);
    return values;
}

template <typename Integer>
[[nodiscard]] Integer parseInteger(std::string_view text, std::string_view option,
                                   Integer minimum, Integer maximum) {
    Integer value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size()
        || value < minimum || value > maximum) {
        throw std::invalid_argument("invalid value for " + std::string(option)
                                  + ": " + std::string(text));
    }
    return value;
}

[[nodiscard]] float parseFloat(std::string_view text, std::string_view option) {
    float value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value,
                                        std::chars_format::general);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size()
        || !std::isfinite(value) || value <= 0.0F) {
        throw std::invalid_argument("invalid value for " + std::string(option)
                                  + ": " + std::string(text));
    }
    return value;
}

[[nodiscard]] Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--allow-software") {
            options.allowSoftware = true;
        } else if (argument == "--mixed-sleep-check") {
            options.mixedSleepCheck = true;
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (argument == "--samples" || argument == "--warmup"
                   || argument == "--prepare-steps"
                   || argument == "--pair-capacity" || argument == "--cell-size") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing value for " + std::string(argument));
            }
            const std::string_view value(argv[++index]);
            if (argument == "--samples") {
                options.samples = parseInteger<std::uint32_t>(
                    value, argument, 1U, 100'000U);
            } else if (argument == "--warmup") {
                options.warmups = parseInteger<std::uint32_t>(
                    value, argument, 0U, 100'000U);
            } else if (argument == "--prepare-steps") {
                options.prepareSteps = parseInteger<std::uint32_t>(
                    value, argument, 0U, 10'000U);
            } else if (argument == "--pair-capacity") {
                options.pairCapacity = parseInteger<std::uint32_t>(
                    value, argument, 1U, std::numeric_limits<std::uint32_t>::max());
            } else {
                options.cellSize = parseFloat(value, argument);
            }
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    return options;
}

void printUsage(std::ostream& stream) {
    stream << "Usage: newtons_gpu_physics_probe [options]\n"
           << "  --samples N          measured fence-blocking submissions (default 12)\n"
           << "  --warmup N           warmup submissions (default 3)\n"
           << "  --prepare-steps N    deterministic CPU preparation ticks (default 8)\n"
           << "  --pair-capacity N    maximum stored candidate pairs (default 1048576)\n"
           << "  --cell-size F        spatial hash cell size in metres (default 0.25)\n"
           << "  --mixed-sleep-check  run active/sleeping compact/large coverage case\n"
           << "  --allow-software     permit llvmpipe/lavapipe/CPU Vulkan devices\n";
}

[[nodiscard]] std::uint32_t bucketCountFor(std::uint32_t bodyCount) {
    std::uint32_t wanted = std::max(256U, bodyCount * 2U);
    std::uint32_t result = 1U;
    while (result < wanted) {
        if (result > (std::numeric_limits<std::uint32_t>::max() >> 1U)) {
            throw std::overflow_error("spatial bucket count overflow");
        }
        result <<= 1U;
    }
    return result;
}

[[nodiscard]] float vectorLength(float x, float y, float z) noexcept {
    return std::sqrt(x * x + y * y + z * z);
}

[[nodiscard]] bool envelopeOverlap(const CpuEnvelope& first,
                                   const CpuEnvelope& second) noexcept {
    const float x = second.center[0] - first.center[0];
    const float y = second.center[1] - first.center[1];
    const float z = second.center[2] - first.center[2];
    const float radius = first.radius + second.radius;
    return x * x + y * y + z * z <= radius * radius;
}

[[nodiscard]] bool trueSweptOverlap(const nec::RigidBody& first,
                                    const nec::RigidBody& second) noexcept {
    const bool firstActive = !first.asleep || first.held;
    const bool secondActive = !second.asleep || second.held;
    nec::Vec3 firstEnd = first.position;
    nec::Vec3 secondEnd = second.position;
    if (firstActive) {
        firstEnd += first.velocity * nec::kFixedDt;
        firstEnd.y -= 0.5 * nec::kEarthGravity
                    * nec::kFixedDt * nec::kFixedDt;
    }
    if (secondActive) {
        secondEnd += second.velocity * nec::kFixedDt;
        secondEnd.y -= 0.5 * nec::kEarthGravity
                     * nec::kFixedDt * nec::kFixedDt;
    }
    const nec::Vec3 start = second.position - first.position;
    const nec::Vec3 motion = (secondEnd - second.position)
                           - (firstEnd - first.position);
    const double startX = start.x;
    const double startY = start.y;
    const double startZ = start.z;
    const double motionX = motion.x;
    const double motionY = motion.y;
    const double motionZ = motion.z;
    const double motionSquared = motionX * motionX + motionY * motionY
                               + motionZ * motionZ;
    double fraction = 0.0;
    if (motionSquared > 1.0e-14) {
        fraction = -(startX * motionX + startY * motionY + startZ * motionZ)
                 / motionSquared;
        fraction = std::clamp(fraction, 0.0, 1.0);
    }
    const double closestX = startX + motionX * fraction;
    const double closestY = startY + motionY * fraction;
    const double closestZ = startZ + motionZ * fraction;
    const double radius = first.cachedBoundingRadius + second.cachedBoundingRadius
                        + nec::kBroadphaseSkin;
    return closestX * closestX + closestY * closestY + closestZ * closestZ
        <= radius * radius;
}

void applyMixedSleepCoverageCase(nec::PhysicsWorld& world, float cellSize) {
    std::vector<std::size_t> compact;
    std::vector<std::size_t> large;
    auto& bodies = world.bodies();
    for (std::size_t index = 0U; index < bodies.size(); ++index) {
        nec::RigidBody& body = bodies[index];
        body.asleep = true;
        body.held = false;
        body.velocity = {};
        const float guardedRadius = static_cast<float>(
            body.cachedBoundingRadius + nec::kBroadphaseSkin * 0.5)
                                  + kRadiusGuard;
        (guardedRadius <= 0.5F * cellSize ? compact : large).push_back(index);
    }
    if (compact.size() < 3U || large.size() < 3U) {
        throw std::runtime_error(
            "mixed-sleep coverage case needs three compact and three large bodies");
    }
    const auto place = [&](std::size_t index, const nec::Vec3& position,
                           bool active) {
        nec::RigidBody& body = bodies[index];
        body.position = position;
        body.previousPosition = position;
        body.velocity = {};
        body.asleep = !active;
    };

    // Inactive-large/active-compact exercises the path that was missing in
    // the first shader draft. Active-large/inactive-compact covers the other
    // direction; the final co-located pair must remain absent because both
    // endpoints sleep.
    place(compact[0], {10.0, 50.0, 10.0}, true);
    place(large[0], {10.0, 50.0, 10.0}, false);
    place(large[1], {35.0, 50.0, 35.0}, true);
    place(compact[1], {35.0, 50.0, 35.0}, false);
    place(compact[2], {60.0, 50.0, 60.0}, false);
    place(large[2], {60.0, 50.0, 60.0}, false);
}

[[nodiscard]] std::vector<InputBody> snapshotWorld(const nec::PhysicsWorld& world) {
    std::vector<InputBody> result;
    result.reserve(world.bodies().size());
    for (const nec::RigidBody& body : world.bodies()) {
        InputBody input;
        const float radius = static_cast<float>(
            body.cachedBoundingRadius + nec::kBroadphaseSkin * 0.5)
                           + kRadiusGuard;
        input.positionRadius = {
            static_cast<float>(body.position.x),
            static_cast<float>(body.position.y),
            static_cast<float>(body.position.z), radius,
        };
        input.previousActive = {
            static_cast<float>(body.previousPosition.x),
            static_cast<float>(body.previousPosition.y),
            static_cast<float>(body.previousPosition.z),
            std::bit_cast<float>((!body.asleep || body.held) ? 1U : 0U),
        };
        input.velocityPad = {
            static_cast<float>(body.velocity.x),
            static_cast<float>(body.velocity.y),
            static_cast<float>(body.velocity.z), 0.0F,
        };
        result.push_back(input);
    }
    return result;
}

[[nodiscard]] CpuReference buildCpuReference(const std::vector<InputBody>& inputs,
                                             const nec::PhysicsWorld& world,
                                             float cellSize) {
    const auto begin = Clock::now();
    std::vector<CpuEnvelope> envelopes;
    envelopes.reserve(inputs.size());
    CpuReference reference;
    const float dt = static_cast<float>(nec::kFixedDt);
    const float gravity = static_cast<float>(nec::kEarthGravity);
    const float gravityStep = 0.5F * gravity * (dt * dt);
    for (const InputBody& input : inputs) {
        CpuEnvelope envelope;
        envelope.active = (std::bit_cast<std::uint32_t>(input.previousActive[3]) & 1U)
                        != 0U;
        envelope.start = {input.positionRadius[0], input.positionRadius[1],
                          input.positionRadius[2]};
        envelope.end = envelope.start;
        if (envelope.active) {
            envelope.end[0] += input.velocityPad[0] * dt;
            envelope.end[1] += input.velocityPad[1] * dt - gravityStep;
            envelope.end[2] += input.velocityPad[2] * dt;
        }
        const float dx = envelope.end[0] - envelope.start[0];
        const float dy = envelope.end[1] - envelope.start[1];
        const float dz = envelope.end[2] - envelope.start[2];
        envelope.center = {
            0.5F * (envelope.start[0] + envelope.end[0]),
            0.5F * (envelope.start[1] + envelope.end[1]),
            0.5F * (envelope.start[2] + envelope.end[2]),
        };
        envelope.baseRadius = std::max(input.positionRadius[3], 0.0F);
        envelope.radius = envelope.baseRadius + 0.5F * vectorLength(dx, dy, dz);
        envelope.compact = envelope.radius <= 0.5F * cellSize;
        if (envelope.compact) {
            ++reference.compactBodies;
        } else {
            ++reference.largeBodies;
        }
        envelopes.push_back(envelope);
    }

    reference.envelopePairs.reserve(16'384U);
    for (std::uint32_t first = 0U;
         first < static_cast<std::uint32_t>(envelopes.size()); ++first) {
        for (std::uint32_t second = first + 1U;
             second < static_cast<std::uint32_t>(envelopes.size()); ++second) {
            if (!envelopes[first].active && !envelopes[second].active) {
                continue;
            }
            const bool envelope = envelopeOverlap(envelopes[first], envelopes[second]);
            const bool exactSweep = trueSweptOverlap(world.bodies()[first],
                                                     world.bodies()[second]);
            if (envelope) {
                reference.envelopePairs.push_back({first, second});
            }
            if (exactSweep) {
                ++reference.trueSweptPairs;
                if (!envelope) {
                    ++reference.trueSubsetMissing;
                }
            }
        }
    }
    if (reference.envelopePairs.size() >= reference.trueSweptPairs) {
        reference.conservativeExtras = reference.envelopePairs.size()
                                     - reference.trueSweptPairs;
    }
    reference.milliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - begin).count();
    return reference;
}

[[nodiscard]] std::vector<std::uint32_t> readShader(std::string_view fileName) {
    namespace fs = std::filesystem;
    std::vector<fs::path> directories;
    std::error_code error;
    const fs::path executable = fs::read_symlink("/proc/self/exe", error);
    if (!error && !executable.empty()) {
        const fs::path binaryDirectory = executable.parent_path();
        directories.push_back(binaryDirectory / "shaders");
        directories.push_back(binaryDirectory / ".." / "share"
                              / "newtons_echo_chamber" / "shaders");
    }
    directories.emplace_back(NEC_SHADER_BINARY_DIR);
    directories.emplace_back(NEC_INSTALL_SHADER_DIR);
    directories.emplace_back("shaders");
    directories.emplace_back("build-vulkan/shaders");
    directories.emplace_back("build/shaders");
    directories.emplace_back("newtons_echo_chamber_cpp/build-vulkan/shaders");

    std::ifstream stream;
    fs::path selected;
    for (const fs::path& directory : directories) {
        const fs::path candidate = directory / fileName;
        stream.open(candidate, std::ios::binary | std::ios::ate);
        if (stream) {
            selected = candidate;
            break;
        }
        stream.clear();
    }
    if (!stream) {
        throw std::runtime_error("cannot find SPIR-V shader " + std::string(fileName));
    }
    const std::streampos end = stream.tellg();
    if (end <= std::streampos{0}) {
        throw std::runtime_error("SPIR-V shader is empty: " + selected.string());
    }
    const auto bytes = static_cast<std::size_t>(end);
    if ((bytes % sizeof(std::uint32_t)) != 0U) {
        throw std::runtime_error("invalid SPIR-V byte count: " + selected.string());
    }
    std::vector<std::uint32_t> words(bytes / sizeof(std::uint32_t));
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char*>(words.data()),
                static_cast<std::streamsize>(bytes));
    if (!stream) {
        throw std::runtime_error("cannot read SPIR-V shader: " + selected.string());
    }
    return words;
}

[[nodiscard]] double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        throw std::invalid_argument("percentile of empty sample set");
    }
    std::sort(values.begin(), values.end());
    const double rank = std::ceil(fraction * static_cast<double>(values.size()));
    const std::size_t index = static_cast<std::size_t>(std::max(1.0, rank)) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

struct ProductionCpuBaseline {
    std::vector<double> milliseconds;
    std::size_t candidates{};
};

// Re-run an identical scene from reset for each sample. PhysicsWorld owns the
// persistent three-worker pool, while its public broadphase timer excludes
// reset, preparation, integration, and contact solving. The extra final tick
// corresponds to the next-tick prediction measured by the GPU probe.
[[nodiscard]] ProductionCpuBaseline benchmarkProductionCpu(
    const Options& options) {
    nec::PhysicsWorld world(1337U, 3U, "auto");
    ProductionCpuBaseline result;
    result.milliseconds.reserve(options.samples);
    const std::uint32_t total = options.warmups + options.samples;
    for (std::uint32_t iteration = 0U; iteration < total; ++iteration) {
        world.reset();
        for (std::uint32_t step = 0U; step <= options.prepareSteps; ++step) {
            world.step(nec::kFixedDt);
        }
        if (iteration >= options.warmups) {
            result.milliseconds.push_back(world.broadphaseMilliseconds());
            const std::size_t candidates = world.broadphaseCandidates();
            if (result.candidates != 0U && result.candidates != candidates) {
                throw std::runtime_error(
                    "production CPU broadphase baseline was not deterministic");
            }
            result.candidates = candidates;
        }
    }
    return result;
}

class VulkanProbe {
public:
    VulkanProbe(const Options& options, std::span<const InputBody> inputs)
        : options_(options), bodyCount_(static_cast<std::uint32_t>(inputs.size())),
          bucketCount_(bucketCountFor(bodyCount_)) {
        try {
            createInstance();
            chooseDevice();
            createDevice();
            createResources(inputs);
            createDescriptorsAndPipelines();
            recordCommands();
        } catch (...) {
            destroy();
            throw;
        }
    }

    VulkanProbe(const VulkanProbe&) = delete;
    VulkanProbe& operator=(const VulkanProbe&) = delete;
    ~VulkanProbe() { destroy(); }

    struct Result {
        std::vector<double> gpuMilliseconds;
        std::vector<double> submitFenceMilliseconds;
        std::vector<double> endToEndMilliseconds;
        std::array<std::uint32_t, 4> counters{};
        std::size_t duplicates{};
        std::size_t missing{};
        std::size_t extra{};
    };

    [[nodiscard]] Result run(const CpuReference& reference,
                             std::span<const InputBody> inputs) {
        if (inputs.size() != bodyCount_) {
            throw std::invalid_argument("GPU input body count changed");
        }
        Result result;
        result.gpuMilliseconds.reserve(options_.samples);
        result.submitFenceMilliseconds.reserve(options_.samples);
        result.endToEndMilliseconds.reserve(options_.samples);
        const std::uint32_t total = options_.warmups + options_.samples;
        for (std::uint32_t iteration = 0U; iteration < total; ++iteration) {
            const auto begin = Clock::now();
            std::memcpy(input_.mapped, inputs.data(), inputs.size_bytes());
            flush(input_);
            requireVk(vkResetFences(device_, 1U, &fence_), "vkResetFences");
            VkSubmitInfo submit = vkInitialize<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
            submit.commandBufferCount = 1U;
            submit.pCommandBuffers = &commandBuffer_;
            const auto queueBegin = Clock::now();
            requireVk(vkQueueSubmit(queue_, 1U, &submit, fence_), "vkQueueSubmit");
            requireVk(vkWaitForFences(device_, 1U, &fence_, VK_TRUE,
                                      std::numeric_limits<std::uint64_t>::max()),
                      "vkWaitForFences");
            const double submitFence = std::chrono::duration<double, std::milli>(
                Clock::now() - queueBegin).count();
            invalidate(counters_);
            invalidate(pairs_);
            std::array<std::uint32_t, 4> counters{};
            std::memcpy(counters.data(), counters_.mapped, sizeof(counters));
            validateResult(reference, counters, result);
            result.counters = counters;
            const double endToEnd = std::chrono::duration<double, std::milli>(
                Clock::now() - begin).count();

            if (iteration >= options_.warmups) {
                result.submitFenceMilliseconds.push_back(submitFence);
                result.endToEndMilliseconds.push_back(endToEnd);
                if (timestamps_) {
                    std::array<std::uint64_t, 2> ticks{};
                    requireVk(vkGetQueryPoolResults(
                                  device_, queryPool_, 0U, 2U, sizeof(ticks),
                                  ticks.data(), sizeof(std::uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                              "vkGetQueryPoolResults");
                    const std::uint64_t mask = timestampValidBits_ >= 64U
                        ? std::numeric_limits<std::uint64_t>::max()
                        : (std::uint64_t{1} << timestampValidBits_) - 1U;
                    const std::uint64_t elapsed = (ticks[1] - ticks[0]) & mask;
                    result.gpuMilliseconds.push_back(
                        static_cast<double>(elapsed) * timestampPeriod_ / 1'000'000.0);
                }
            }
        }
        return result;
    }

    [[nodiscard]] std::string_view deviceName() const noexcept {
        return properties_.deviceName;
    }
    [[nodiscard]] std::uint32_t apiVersion() const noexcept {
        return properties_.apiVersion;
    }
    [[nodiscard]] bool core13() const noexcept { return core13_; }
    [[nodiscard]] bool software() const noexcept { return software_; }
    [[nodiscard]] bool timestamps() const noexcept { return timestamps_; }
    [[nodiscard]] std::uint32_t bucketCount() const noexcept { return bucketCount_; }
    [[nodiscard]] bool allBuffersCoherent() const noexcept {
        return input_.coherent && broadphase_.coherent && buckets_.coherent
            && next_.coherent && counters_.coherent && pairs_.coherent;
    }
    [[nodiscard]] bool allBuffersDeviceLocal() const noexcept {
        return input_.deviceLocal && broadphase_.deviceLocal && buckets_.deviceLocal
            && next_.deviceLocal && counters_.deviceLocal && pairs_.deviceLocal;
    }
    [[nodiscard]] VkDeviceSize totalBufferBytes() const noexcept {
        return input_.size + broadphase_.size + buckets_.size + next_.size
             + counters_.size + pairs_.size;
    }

private:
    void createInstance() {
        std::uint32_t loader = VK_API_VERSION_1_0;
        requireVk(vkEnumerateInstanceVersion(&loader), "vkEnumerateInstanceVersion");
        if (loader < VK_API_VERSION_1_2) {
            throw std::runtime_error("Vulkan loader 1.2 or newer is required");
        }
        instanceApi_ = std::min(loader, VK_API_VERSION_1_3);
        VkApplicationInfo application = vkInitialize<VkApplicationInfo>(
            VK_STRUCTURE_TYPE_APPLICATION_INFO);
        application.pApplicationName = "Newton's Echo Chamber GPU physics probe";
        application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
        application.pEngineName = "NEC";
        application.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
        application.apiVersion = instanceApi_;
        VkInstanceCreateInfo create = vkInitialize<VkInstanceCreateInfo>(
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
        create.pApplicationInfo = &application;
        requireVk(vkCreateInstance(&create, nullptr, &instance_), "vkCreateInstance");
    }

    void chooseDevice() {
        std::uint32_t count = 0U;
        requireVk(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
                  "vkEnumeratePhysicalDevices(count)");
        if (count == 0U) {
            throw std::runtime_error("no Vulkan physical device found");
        }
        std::vector<VkPhysicalDevice> devices(count);
        requireVk(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
                  "vkEnumeratePhysicalDevices");
        devices.resize(count);
        std::optional<DeviceCandidate> best;
        for (VkPhysicalDevice device : devices) {
            DeviceCandidate candidate;
            candidate.device = device;
            vkGetPhysicalDeviceProperties(device, &candidate.properties);
            candidate.software = isSoftware(candidate.properties);
            if (candidate.software && !options_.allowSoftware) {
                continue;
            }
            if (candidate.properties.apiVersion < VK_API_VERSION_1_2) {
                continue;
            }
            const auto extensions = deviceExtensions(device);
            candidate.core13 = instanceApi_ >= VK_API_VERSION_1_3
                            && candidate.properties.apiVersion >= VK_API_VERSION_1_3;
            if (!candidate.core13
                && !hasExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)) {
                continue;
            }

            bool synchronization2 = false;
            VkPhysicalDeviceFeatures2 features = vkInitialize<VkPhysicalDeviceFeatures2>(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
            VkPhysicalDeviceVulkan13Features features13 =
                vkInitialize<VkPhysicalDeviceVulkan13Features>(
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
            VkPhysicalDeviceSynchronization2FeaturesKHR featuresKhr =
                vkInitialize<VkPhysicalDeviceSynchronization2FeaturesKHR>(
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR);
            features.pNext = candidate.core13
                ? static_cast<void*>(&features13) : static_cast<void*>(&featuresKhr);
            vkGetPhysicalDeviceFeatures2(device, &features);
            synchronization2 = candidate.core13
                ? features13.synchronization2 == VK_TRUE
                : featuresKhr.synchronization2 == VK_TRUE;
            if (!synchronization2) {
                continue;
            }

            std::uint32_t queueCount = 0U;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, nullptr);
            std::vector<VkQueueFamilyProperties> queues(queueCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, queues.data());
            int bestQueueScore = -1;
            for (std::uint32_t family = 0U; family < queueCount; ++family) {
                if (queues[family].queueCount == 0U
                    || (queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0U) {
                    continue;
                }
                int score = 1;
                if ((queues[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0U) {
                    score += 2;
                }
                if (score > bestQueueScore) {
                    bestQueueScore = score;
                    candidate.queueFamily = family;
                    candidate.timestampValidBits = queues[family].timestampValidBits;
                }
            }
            if (candidate.queueFamily == std::numeric_limits<std::uint32_t>::max()) {
                continue;
            }
            candidate.score = isV3dv(candidate.properties.deviceName)
                && !candidate.software ? 1'000'000ULL : 0ULL;
            switch (candidate.properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                candidate.score += 50'000ULL;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                candidate.score += 40'000ULL;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                candidate.score += 10'000ULL;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_OTHER:
                candidate.score += 5'000ULL;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                candidate.score += 1ULL;
                break;
            default:
                break;
            }
            if (!best || candidate.score > best->score) {
                best = candidate;
            }
        }
        if (!best) {
            throw std::runtime_error(
                "no acceptable Vulkan compute device with synchronization2; "
                "software devices are rejected unless --allow-software is supplied");
        }
        physicalDevice_ = best->device;
        properties_ = best->properties;
        queueFamily_ = best->queueFamily;
        timestampValidBits_ = best->timestampValidBits;
        core13_ = best->core13;
        software_ = best->software;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties_);
        timestampPeriod_ = static_cast<double>(properties_.limits.timestampPeriod);
        timestamps_ = timestampValidBits_ != 0U
                   && properties_.limits.timestampComputeAndGraphics == VK_TRUE
                   && timestampPeriod_ > 0.0;
    }

    void createDevice() {
        constexpr float priority = 1.0F;
        VkDeviceQueueCreateInfo queueCreate = vkInitialize<VkDeviceQueueCreateInfo>(
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
        queueCreate.queueFamilyIndex = queueFamily_;
        queueCreate.queueCount = 1U;
        queueCreate.pQueuePriorities = &priority;
        VkPhysicalDeviceVulkan13Features features13 =
            vkInitialize<VkPhysicalDeviceVulkan13Features>(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        VkPhysicalDeviceSynchronization2FeaturesKHR featuresKhr =
            vkInitialize<VkPhysicalDeviceSynchronization2FeaturesKHR>(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR);
        if (core13_) {
            features13.synchronization2 = VK_TRUE;
        } else {
            featuresKhr.synchronization2 = VK_TRUE;
        }
        const char* extension = VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME;
        VkDeviceCreateInfo create = vkInitialize<VkDeviceCreateInfo>(
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
        create.pNext = core13_ ? static_cast<void*>(&features13)
                              : static_cast<void*>(&featuresKhr);
        create.queueCreateInfoCount = 1U;
        create.pQueueCreateInfos = &queueCreate;
        if (!core13_) {
            create.enabledExtensionCount = 1U;
            create.ppEnabledExtensionNames = &extension;
        }
        requireVk(vkCreateDevice(physicalDevice_, &create, nullptr, &device_),
                  "vkCreateDevice");
        vkGetDeviceQueue(device_, queueFamily_, 0U, &queue_);
        pipelineBarrier2_ = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(
            vkGetDeviceProcAddr(device_, "vkCmdPipelineBarrier2"));
        if (pipelineBarrier2_ == nullptr) {
            pipelineBarrier2_ = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(
                vkGetDeviceProcAddr(device_, "vkCmdPipelineBarrier2KHR"));
        }
        if (pipelineBarrier2_ == nullptr) {
            throw std::runtime_error("vkCmdPipelineBarrier2/KHR is unavailable");
        }
    }

    [[nodiscard]] std::pair<std::uint32_t, bool>
    findMemoryType(std::uint32_t bits) const {
        std::optional<std::pair<std::uint32_t, bool>> fallback;
        for (std::uint32_t index = 0U;
             index < memoryProperties_.memoryTypeCount; ++index) {
            if ((bits & (1U << index)) == 0U) {
                continue;
            }
            const VkMemoryPropertyFlags flags =
                memoryProperties_.memoryTypes[index].propertyFlags;
            if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0U) {
                continue;
            }
            const bool coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U;
            if (coherent && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0U) {
                return {index, true};
            }
            if (!fallback || coherent) {
                fallback = std::pair<std::uint32_t, bool>{index, coherent};
            }
        }
        if (!fallback) {
            throw std::runtime_error("no host-visible Vulkan memory type for buffer");
        }
        return *fallback;
    }

    void createBuffer(Buffer& buffer, VkDeviceSize size) {
        buffer.size = size;
        VkBufferCreateInfo create = vkInitialize<VkBufferCreateInfo>(
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        create.size = size;
        create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        requireVk(vkCreateBuffer(device_, &create, nullptr, &buffer.handle),
                  "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer.handle, &requirements);
        const auto [type, coherent] = findMemoryType(requirements.memoryTypeBits);
        VkMemoryAllocateInfo allocation = vkInitialize<VkMemoryAllocateInfo>(
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        requireVk(vkAllocateMemory(device_, &allocation, nullptr, &buffer.memory),
                  "vkAllocateMemory");
        requireVk(vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0U),
                  "vkBindBufferMemory");
        requireVk(vkMapMemory(device_, buffer.memory, 0U, VK_WHOLE_SIZE, 0U,
                              &buffer.mapped), "vkMapMemory");
        buffer.allocationSize = requirements.size;
        buffer.coherent = coherent;
        buffer.deviceLocal =
            (memoryProperties_.memoryTypes[type].propertyFlags
             & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0U;
    }

    void flush(const Buffer& buffer) const {
        if (buffer.coherent) {
            return;
        }
        VkMappedMemoryRange range = vkInitialize<VkMappedMemoryRange>(
            VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
        range.memory = buffer.memory;
        range.offset = 0U;
        range.size = VK_WHOLE_SIZE;
        requireVk(vkFlushMappedMemoryRanges(device_, 1U, &range),
                  "vkFlushMappedMemoryRanges");
    }

    void invalidate(const Buffer& buffer) const {
        if (buffer.coherent) {
            return;
        }
        VkMappedMemoryRange range = vkInitialize<VkMappedMemoryRange>(
            VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
        range.memory = buffer.memory;
        range.offset = 0U;
        range.size = VK_WHOLE_SIZE;
        requireVk(vkInvalidateMappedMemoryRanges(device_, 1U, &range),
                  "vkInvalidateMappedMemoryRanges");
    }

    void createResources(std::span<const InputBody> inputs) {
        const VkDeviceSize inputBytes = static_cast<VkDeviceSize>(inputs.size_bytes());
        const VkDeviceSize broadphaseBytes = static_cast<VkDeviceSize>(bodyCount_)
                                              * sizeof(BroadphaseBody);
        const VkDeviceSize bucketBytes = static_cast<VkDeviceSize>(bucketCount_)
                                          * sizeof(std::int32_t);
        const VkDeviceSize nextBytes = static_cast<VkDeviceSize>(bodyCount_)
                                        * sizeof(std::int32_t);
        const VkDeviceSize pairBytes = static_cast<VkDeviceSize>(options_.pairCapacity)
                                        * sizeof(GpuPair);
        if (inputBytes > properties_.limits.maxStorageBufferRange
            || broadphaseBytes > properties_.limits.maxStorageBufferRange
            || bucketBytes > properties_.limits.maxStorageBufferRange
            || nextBytes > properties_.limits.maxStorageBufferRange
            || pairBytes > properties_.limits.maxStorageBufferRange) {
            throw std::runtime_error("requested buffer exceeds maxStorageBufferRange");
        }
        createBuffer(input_, inputBytes);
        createBuffer(broadphase_, broadphaseBytes);
        createBuffer(buckets_, bucketBytes);
        createBuffer(next_, nextBytes);
        createBuffer(counters_, sizeof(std::uint32_t) * 4U);
        createBuffer(pairs_, pairBytes);
        std::memcpy(input_.mapped, inputs.data(), inputs.size_bytes());
        flush(input_);

        VkCommandPoolCreateInfo pool = vkInitialize<VkCommandPoolCreateInfo>(
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
        pool.queueFamilyIndex = queueFamily_;
        requireVk(vkCreateCommandPool(device_, &pool, nullptr, &commandPool_),
                  "vkCreateCommandPool");
        VkCommandBufferAllocateInfo allocate =
            vkInitialize<VkCommandBufferAllocateInfo>(
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
        allocate.commandPool = commandPool_;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1U;
        requireVk(vkAllocateCommandBuffers(device_, &allocate, &commandBuffer_),
                  "vkAllocateCommandBuffers");
        VkFenceCreateInfo fence = vkInitialize<VkFenceCreateInfo>(
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        requireVk(vkCreateFence(device_, &fence, nullptr, &fence_), "vkCreateFence");
        if (timestamps_) {
            VkQueryPoolCreateInfo query = vkInitialize<VkQueryPoolCreateInfo>(
                VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
            query.queryType = VK_QUERY_TYPE_TIMESTAMP;
            query.queryCount = 2U;
            requireVk(vkCreateQueryPool(device_, &query, nullptr, &queryPool_),
                      "vkCreateQueryPool");
        }
    }

    [[nodiscard]] VkShaderModule createShader(std::string_view fileName) const {
        const std::vector<std::uint32_t> words = readShader(fileName);
        VkShaderModuleCreateInfo create = vkInitialize<VkShaderModuleCreateInfo>(
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        create.codeSize = words.size() * sizeof(std::uint32_t);
        create.pCode = words.data();
        VkShaderModule module = VK_NULL_HANDLE;
        requireVk(vkCreateShaderModule(device_, &create, nullptr, &module),
                  "vkCreateShaderModule");
        return module;
    }

    [[nodiscard]] VkDescriptorSetLayout createSetLayout() const {
        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
        for (std::uint32_t index = 0U; index < bindings.size(); ++index) {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1U;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo create =
            vkInitialize<VkDescriptorSetLayoutCreateInfo>(
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
        create.bindingCount = static_cast<std::uint32_t>(bindings.size());
        create.pBindings = bindings.data();
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        requireVk(vkCreateDescriptorSetLayout(device_, &create, nullptr, &layout),
                  "vkCreateDescriptorSetLayout");
        return layout;
    }

    [[nodiscard]] VkPipelineLayout createPipelineLayout(
        VkDescriptorSetLayout setLayout, std::uint32_t pushSize) const {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = pushSize;
        VkPipelineLayoutCreateInfo create = vkInitialize<VkPipelineLayoutCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
        create.setLayoutCount = 1U;
        create.pSetLayouts = &setLayout;
        create.pushConstantRangeCount = 1U;
        create.pPushConstantRanges = &push;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        requireVk(vkCreatePipelineLayout(device_, &create, nullptr, &layout),
                  "vkCreatePipelineLayout");
        return layout;
    }

    [[nodiscard]] VkPipeline createPipeline(VkPipelineLayout layout,
                                            VkShaderModule shader) const {
        VkPipelineShaderStageCreateInfo stage =
            vkInitialize<VkPipelineShaderStageCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        VkComputePipelineCreateInfo create =
            vkInitialize<VkComputePipelineCreateInfo>(
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
        create.stage = stage;
        create.layout = layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        requireVk(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1U, &create,
                                           nullptr, &pipeline),
                  "vkCreateComputePipelines");
        return pipeline;
    }

    void updateSet(VkDescriptorSet set, const std::array<Buffer*, 5>& buffers) {
        std::array<VkDescriptorBufferInfo, 5> infos{};
        std::array<VkWriteDescriptorSet, 5> writes{};
        for (std::uint32_t index = 0U; index < buffers.size(); ++index) {
            infos[index].buffer = buffers[index]->handle;
            infos[index].range = buffers[index]->size;
            writes[index] = vkInitialize<VkWriteDescriptorSet>(
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
            writes[index].dstSet = set;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1U;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index].pBufferInfo = &infos[index];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0U, nullptr);
    }

    void createDescriptorsAndPipelines() {
        integrateSetLayout_ = createSetLayout();
        broadphaseSetLayout_ = createSetLayout();
        integrateLayout_ = createPipelineLayout(
            integrateSetLayout_, static_cast<std::uint32_t>(sizeof(IntegratePush)));
        broadphaseLayout_ = createPipelineLayout(
            broadphaseSetLayout_, static_cast<std::uint32_t>(sizeof(BroadphasePush)));
        VkShaderModule integrateShader = VK_NULL_HANDLE;
        VkShaderModule broadphaseShader = VK_NULL_HANDLE;
        try {
            integrateShader = createShader("rigid_integrate_hash.comp.spv");
            broadphaseShader = createShader("rigid_broadphase.comp.spv");
            integratePipeline_ = createPipeline(integrateLayout_, integrateShader);
            broadphasePipeline_ = createPipeline(broadphaseLayout_, broadphaseShader);
        } catch (...) {
            if (integrateShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, integrateShader, nullptr);
            }
            if (broadphaseShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, broadphaseShader, nullptr);
            }
            throw;
        }
        vkDestroyShaderModule(device_, integrateShader, nullptr);
        vkDestroyShaderModule(device_, broadphaseShader, nullptr);

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 10U;
        VkDescriptorPoolCreateInfo pool = vkInitialize<VkDescriptorPoolCreateInfo>(
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
        pool.maxSets = 2U;
        pool.poolSizeCount = 1U;
        pool.pPoolSizes = &poolSize;
        requireVk(vkCreateDescriptorPool(device_, &pool, nullptr, &descriptorPool_),
                  "vkCreateDescriptorPool");
        std::array<VkDescriptorSetLayout, 2> layouts{
            integrateSetLayout_, broadphaseSetLayout_};
        std::array<VkDescriptorSet, 2> sets{};
        VkDescriptorSetAllocateInfo allocate =
            vkInitialize<VkDescriptorSetAllocateInfo>(
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
        allocate.descriptorPool = descriptorPool_;
        allocate.descriptorSetCount = static_cast<std::uint32_t>(layouts.size());
        allocate.pSetLayouts = layouts.data();
        requireVk(vkAllocateDescriptorSets(device_, &allocate, sets.data()),
                  "vkAllocateDescriptorSets");
        integrateSet_ = sets[0];
        broadphaseSet_ = sets[1];
        updateSet(integrateSet_, {&input_, &broadphase_, &buckets_, &next_,
                                  &counters_});
        updateSet(broadphaseSet_, {&broadphase_, &buckets_, &next_, &counters_,
                                   &pairs_});
    }

    void barrier(VkPipelineStageFlags2 sourceStage, VkAccessFlags2 sourceAccess,
                 VkPipelineStageFlags2 destinationStage,
                 VkAccessFlags2 destinationAccess) const {
        VkMemoryBarrier2 memory = vkInitialize<VkMemoryBarrier2>(
            VK_STRUCTURE_TYPE_MEMORY_BARRIER_2);
        memory.srcStageMask = sourceStage;
        memory.srcAccessMask = sourceAccess;
        memory.dstStageMask = destinationStage;
        memory.dstAccessMask = destinationAccess;
        VkDependencyInfo dependency = vkInitialize<VkDependencyInfo>(
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        dependency.memoryBarrierCount = 1U;
        dependency.pMemoryBarriers = &memory;
        pipelineBarrier2_(commandBuffer_, &dependency);
    }

    void recordCommands() {
        VkCommandBufferBeginInfo begin = vkInitialize<VkCommandBufferBeginInfo>(
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        requireVk(vkBeginCommandBuffer(commandBuffer_, &begin), "vkBeginCommandBuffer");
        if (timestamps_) {
            vkCmdResetQueryPool(commandBuffer_, queryPool_, 0U, 2U);
        }
        vkCmdFillBuffer(commandBuffer_, buckets_.handle, 0U, VK_WHOLE_SIZE,
                        std::numeric_limits<std::uint32_t>::max());
        vkCmdFillBuffer(commandBuffer_, next_.handle, 0U, VK_WHOLE_SIZE,
                        std::numeric_limits<std::uint32_t>::max());
        vkCmdFillBuffer(commandBuffer_, counters_.handle, 0U, VK_WHOLE_SIZE, 0U);
        barrier(VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                    | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        if (timestamps_) {
            vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                queryPool_, 0U);
        }

        const std::uint32_t groups = (bodyCount_ + kLocalSize - 1U) / kLocalSize;
        const IntegratePush integratePush{
            bodyCount_, static_cast<float>(nec::kFixedDt),
            static_cast<float>(nec::kEarthGravity), options_.cellSize,
            bucketCount_ - 1U,
        };
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                          integratePipeline_);
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                integrateLayout_, 0U, 1U, &integrateSet_, 0U, nullptr);
        vkCmdPushConstants(commandBuffer_, integrateLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<std::uint32_t>(sizeof(integratePush)),
                           &integratePush);
        vkCmdDispatch(commandBuffer_, groups, 1U, 1U);
        barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                    | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        const BroadphasePush broadphasePush{
            bodyCount_, bucketCount_ - 1U, options_.pairCapacity};
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                          broadphasePipeline_);
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                broadphaseLayout_, 0U, 1U, &broadphaseSet_, 0U, nullptr);
        vkCmdPushConstants(commandBuffer_, broadphaseLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<std::uint32_t>(sizeof(broadphasePush)),
                           &broadphasePush);
        vkCmdDispatch(commandBuffer_, groups, 1U, 1U);
        if (timestamps_) {
            vkCmdWriteTimestamp(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                queryPool_, 1U);
        }
        barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        requireVk(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer");
    }

    void validateResult(const CpuReference& reference,
                        const std::array<std::uint32_t, 4>& counters,
                        Result& aggregate) const {
        if (counters[1] != 0U || counters[0] > options_.pairCapacity) {
            throw std::runtime_error(
                "GPU pair buffer overflow: attempted=" + std::to_string(counters[0])
                + " dropped=" + std::to_string(counters[1])
                + " capacity=" + std::to_string(options_.pairCapacity));
        }
        if (counters[2] != reference.compactBodies
            || counters[3] != reference.largeBodies) {
            throw std::runtime_error(
                "GPU compact/large classification differs from CPU reference");
        }
        std::vector<GpuPair> gpuPairs(counters[0]);
        std::memcpy(gpuPairs.data(), pairs_.mapped,
                    gpuPairs.size() * sizeof(GpuPair));
        for (GpuPair& pair : gpuPairs) {
            if (pair.second < pair.first) {
                std::swap(pair.first, pair.second);
            }
            if (pair.first == pair.second || pair.second >= bodyCount_) {
                throw std::runtime_error("GPU emitted invalid body pair");
            }
        }
        std::sort(gpuPairs.begin(), gpuPairs.end());
        const auto uniqueEnd = std::unique(gpuPairs.begin(), gpuPairs.end());
        const std::size_t duplicates = static_cast<std::size_t>(
            std::distance(uniqueEnd, gpuPairs.end()));
        gpuPairs.erase(uniqueEnd, gpuPairs.end());

        std::size_t missing = 0U;
        std::size_t extra = 0U;
        std::size_t gpuIndex = 0U;
        std::size_t cpuIndex = 0U;
        while (gpuIndex < gpuPairs.size()
               && cpuIndex < reference.envelopePairs.size()) {
            if (gpuPairs[gpuIndex] < reference.envelopePairs[cpuIndex]) {
                ++extra;
                ++gpuIndex;
            } else if (reference.envelopePairs[cpuIndex] < gpuPairs[gpuIndex]) {
                ++missing;
                ++cpuIndex;
            } else {
                ++gpuIndex;
                ++cpuIndex;
            }
        }
        extra += gpuPairs.size() - gpuIndex;
        missing += reference.envelopePairs.size() - cpuIndex;
        aggregate.duplicates = std::max(aggregate.duplicates, duplicates);
        aggregate.missing = std::max(aggregate.missing, missing);
        aggregate.extra = std::max(aggregate.extra, extra);
        if (duplicates != 0U || missing != 0U || extra != 0U) {
            throw std::runtime_error(
                "GPU pair verification failed: duplicates="
                + std::to_string(duplicates) + " missing=" + std::to_string(missing)
                + " extra=" + std::to_string(extra));
        }
    }

    void destroyBuffer(Buffer& buffer) noexcept {
        if (device_ == VK_NULL_HANDLE) {
            return;
        }
        if (buffer.mapped != nullptr) {
            vkUnmapMemory(device_, buffer.memory);
            buffer.mapped = nullptr;
        }
        if (buffer.handle != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer.handle, nullptr);
            buffer.handle = VK_NULL_HANDLE;
        }
        if (buffer.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, buffer.memory, nullptr);
            buffer.memory = VK_NULL_HANDLE;
        }
    }

    void destroy() noexcept {
        if (device_ != VK_NULL_HANDLE) {
            (void)vkDeviceWaitIdle(device_);
            if (queryPool_ != VK_NULL_HANDLE) {
                vkDestroyQueryPool(device_, queryPool_, nullptr);
            }
            if (fence_ != VK_NULL_HANDLE) {
                vkDestroyFence(device_, fence_, nullptr);
            }
            if (commandPool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, commandPool_, nullptr);
            }
            if (descriptorPool_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
            }
            if (integratePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, integratePipeline_, nullptr);
            }
            if (broadphasePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, broadphasePipeline_, nullptr);
            }
            if (integrateLayout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, integrateLayout_, nullptr);
            }
            if (broadphaseLayout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, broadphaseLayout_, nullptr);
            }
            if (integrateSetLayout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, integrateSetLayout_, nullptr);
            }
            if (broadphaseSetLayout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, broadphaseSetLayout_, nullptr);
            }
            destroyBuffer(pairs_);
            destroyBuffer(counters_);
            destroyBuffer(next_);
            destroyBuffer(buckets_);
            destroyBuffer(broadphase_);
            destroyBuffer(input_);
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
    }

    Options options_;
    std::uint32_t bodyCount_{};
    std::uint32_t bucketCount_{};
    std::uint32_t instanceApi_{};
    VkInstance instance_{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice_{VK_NULL_HANDLE};
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    std::uint32_t queueFamily_{};
    std::uint32_t timestampValidBits_{};
    double timestampPeriod_{};
    bool core13_{};
    bool software_{};
    bool timestamps_{};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueue queue_{VK_NULL_HANDLE};
    PFN_vkCmdPipelineBarrier2 pipelineBarrier2_{};
    Buffer input_{};
    Buffer broadphase_{};
    Buffer buckets_{};
    Buffer next_{};
    Buffer counters_{};
    Buffer pairs_{};
    VkDescriptorSetLayout integrateSetLayout_{VK_NULL_HANDLE};
    VkDescriptorSetLayout broadphaseSetLayout_{VK_NULL_HANDLE};
    VkPipelineLayout integrateLayout_{VK_NULL_HANDLE};
    VkPipelineLayout broadphaseLayout_{VK_NULL_HANDLE};
    VkPipeline integratePipeline_{VK_NULL_HANDLE};
    VkPipeline broadphasePipeline_{VK_NULL_HANDLE};
    VkDescriptorPool descriptorPool_{VK_NULL_HANDLE};
    VkDescriptorSet integrateSet_{VK_NULL_HANDLE};
    VkDescriptorSet broadphaseSet_{VK_NULL_HANDLE};
    VkCommandPool commandPool_{VK_NULL_HANDLE};
    VkCommandBuffer commandBuffer_{VK_NULL_HANDLE};
    VkFence fence_{VK_NULL_HANDLE};
    VkQueryPool queryPool_{VK_NULL_HANDLE};
};

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.help) {
            printUsage(std::cout);
            return 0;
        }

        nec::PhysicsWorld world(1337U, 3U, "auto");
        for (std::uint32_t step = 0U; step < options.prepareSteps; ++step) {
            world.step(nec::kFixedDt);
        }
        if (options.mixedSleepCheck) {
            applyMixedSleepCoverageCase(world, options.cellSize);
        }
        const std::vector<InputBody> inputs = snapshotWorld(world);
        if (inputs.size() != static_cast<std::size_t>(nec::kExpectedBodyCount)) {
            throw std::runtime_error("physics snapshot does not contain 4,487 bodies");
        }
        const std::size_t activeCount = static_cast<std::size_t>(std::count_if(
            inputs.begin(), inputs.end(), [](const InputBody& input) {
                return (std::bit_cast<std::uint32_t>(input.previousActive[3]) & 1U)
                    != 0U;
            }));
        const CpuReference reference = buildCpuReference(inputs, world,
                                                         options.cellSize);
        if (reference.trueSubsetMissing != 0U) {
            throw std::runtime_error(
                "CPU envelope failed to contain the true swept-sphere reference");
        }
        if (options.mixedSleepCheck
            && (activeCount != 2U || reference.envelopePairs.size() != 2U
                || reference.trueSweptPairs != 2U)) {
            throw std::runtime_error(
                "mixed-sleep coverage case did not produce its two expected pairs");
        }
        const std::optional<ProductionCpuBaseline> productionCpu =
            options.mixedSleepCheck
                ? std::nullopt
                : std::optional<ProductionCpuBaseline>{
                      benchmarkProductionCpu(options)};

        VulkanProbe probe(options, inputs);
        const VulkanProbe::Result result = probe.run(reference, inputs);

        std::cout << std::fixed << std::setprecision(3)
                  << "device=" << probe.deviceName()
                  << " api=" << versionString(probe.apiVersion())
                  << " synchronization2=" << (probe.core13() ? "core" : "KHR")
                  << " software=" << (probe.software() ? "yes" : "no") << '\n'
                  << "bodies=" << inputs.size() << " active=" << activeCount
                  << " compact=" << result.counters[2]
                  << " large=" << result.counters[3]
                  << " envelope_pairs=" << reference.envelopePairs.size()
                  << " true_swept_pairs=" << reference.trueSweptPairs
                  << " conservative_extras=" << reference.conservativeExtras << '\n'
                  << "prepare_steps=" << options.prepareSteps
                  << " validation_case="
                  << (options.mixedSleepCheck ? "mixed-sleep" : "scene")
                  << " cell_size=" << options.cellSize
                  << " buckets=" << probe.bucketCount()
                  << " pair_capacity=" << options.pairCapacity << '\n'
                  << "host_memory="
                  << (probe.allBuffersCoherent() ? "coherent" : "noncoherent-or-mixed")
                  << " device_local="
                  << (probe.allBuffersDeviceLocal() ? "yes" : "no-or-mixed")
                  << " buffer_mib="
                  << static_cast<double>(probe.totalBufferBytes())
                     / (1024.0 * 1024.0) << '\n'
                  << "cpu_all_pairs_oracle_ms=" << reference.milliseconds << '\n';
        if (productionCpu) {
            std::cout << "cpu_production_broadphase_ms median="
                      << percentile(productionCpu->milliseconds, 0.50)
                      << " p95=" << percentile(productionCpu->milliseconds, 0.95)
                      << " candidates=" << productionCpu->candidates << '\n';
        } else {
            std::cout << "cpu_production_broadphase_ms unavailable_for_synthetic_case\n";
        }
        if (probe.timestamps()) {
            std::cout << "gpu_timestamp_ms median="
                      << percentile(result.gpuMilliseconds, 0.50)
                      << " p95=" << percentile(result.gpuMilliseconds, 0.95) << '\n';
        } else {
            std::cout << "gpu_timestamp_ms unavailable\n";
        }
        std::cout << "submit_fence_ms median="
                  << percentile(result.submitFenceMilliseconds, 0.50)
                  << " p95=" << percentile(result.submitFenceMilliseconds, 0.95)
                  << '\n'
                  << "verified_end_to_end_ms median="
                  << percentile(result.endToEndMilliseconds, 0.50)
                  << " p95=" << percentile(result.endToEndMilliseconds, 0.95) << '\n'
                  << "samples=" << options.samples
                  << " warmup=" << options.warmups << '\n'
                  << "verification=PASS overflow=0 duplicates=" << result.duplicates
                  << " missing=" << result.missing << " extra=" << result.extra << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gpu physics probe: " << error.what() << '\n';
        return 1;
    }
}
