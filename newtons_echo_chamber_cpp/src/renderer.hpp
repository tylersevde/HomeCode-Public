#pragma once

#include "physics.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

namespace nec {

class Platform;

struct GpuPhysicsStats {
    bool available{};
    std::uint32_t activeParticles{};
    std::uint32_t capacity{};
    std::uint64_t storageBytes{};
    std::uint64_t simulatedFrames{};
};

struct VulkanStats {
    std::string_view deviceName;
    std::string_view driverName;
    std::uint32_t deviceApiVersion{};
    std::uint32_t loaderApiVersion{};
    std::uint64_t deviceHeapBytes{};
    std::uint64_t deviceBudgetBytes{};
    std::uint64_t deviceUsageBytes{};
    std::uint32_t queueFamily{};
    std::uint32_t swapchainImages{};
    std::uint64_t renderedFrames{};
    std::uint32_t validationErrors{};
    bool coreVulkan13{};
    bool dynamicRendering{};
    bool synchronization2{};
    bool timelineSemaphore{};
    bool memoryBudget{};
    bool softwareDevice{};
    bool flashlightShadowsAvailable{};
    bool flashlightShadowsEnabled{};
    std::uint64_t shadowedFrames{};
    bool gpuTimingAvailable{};
    std::uint64_t gpuTimedFrames{};
    double averageGpuFrameMilliseconds{};
    double averageGpuComputeMilliseconds{};
    double averageGpuShadowMilliseconds{};
    double averageGpuGraphicsMilliseconds{};
};

// Small, copyable presentation state supplied by the application once per
// frame. The renderer derives all world/probe values directly from
// PhysicsWorld so callers do not need to duplicate physics bookkeeping.
struct RendererHudState {
    bool visible{true};
    bool helpVisible{};
    bool paused{};
    bool singleStep{};
    double fps{};
    int quality{1};
    bool echoVisible{true};
    bool kineticLens{};
    bool resonanceVisible{true};
    // F12 presentation policy: true fades/skips distant normal-map samples
    // using the quality-tier ranges; false retains full normal detail.
    bool adaptiveNormalDetail{true};
    bool audioAvailable{true};
    bool audioMuted{};
};

// Vulkan 1.3 feature-profile renderer. Raspberry Pi's V3DV currently exposes
// Vulkan 1.2, so promoted KHR dynamic-rendering and synchronization2 entry
// points are negotiated when the equivalent core 1.3 commands are unavailable.
class Renderer final {
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    [[nodiscard]] bool initialize(const Platform& platform, std::uint32_t seed,
                                  bool maximumGpuPhysics = false,
                                  bool enableValidation = false,
                                  bool allowSoftwareDevice = false);
    void shutdown() noexcept;
    void waitIdle() noexcept;
    void resetPerformanceTimings() noexcept;

    void setQuality(int quality) noexcept;
    void setMaximumGpuPhysics(bool enabled) noexcept;
    void setGpuPhysicsEnabled(bool enabled) noexcept;
    void setOverheadLightsEnabled(bool enabled) noexcept;
    void setFlashlightEnabled(bool enabled) noexcept;
    void setFlashlightShadowsEnabled(bool enabled) noexcept;
    void setMaterialTexturesEnabled(bool enabled) noexcept;
    void setNormalMappingEnabled(bool enabled) noexcept;
    void setHudState(const RendererHudState& state) noexcept;

    // Records compute and graphics into one ordered V3D queue submission. CPU
    // simulation remains concurrent with already-submitted GPU work; no CPU
    // readback occurs on the frame path.
    [[nodiscard]] bool render(const PhysicsWorld& world,
                              float interpolationAlpha, float frameDeltaSeconds);

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] bool validationClean() const noexcept;
    [[nodiscard]] bool deviceLost() const noexcept;
    [[nodiscard]] std::string_view lastError() const noexcept;
    [[nodiscard]] GpuPhysicsStats gpuPhysicsStats() const noexcept;
    [[nodiscard]] VulkanStats vulkanStats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nec
