#include "audio.hpp"
#include "physics.hpp"
#include "platform.hpp"
#include "renderer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace nec {
namespace {

class FrameGovernor final {
public:
    void observe(double seconds) noexcept {
        const double sample = clamp(seconds, 1.0 / 500.0, kMaxFrameDt);
        const double alpha = 1.0 - std::exp(-sample / 0.35);
        emaSeconds_ += (sample - emaSeconds_) * alpha;
        const double fps = 1.0 / std::max(1.0e-6, emaSeconds_);
        const int desired = fps < 55.0 ? 0 : fps < 59.0 ? 1 : 2;

        if (desired < quality_) {
            recoveryAge_ = 0.0;
            slowAge_ += sample;
            const double delay = desired == 0 ? 0.10 : 0.55;
            if (slowAge_ >= delay) {
                quality_ = desired == 0 ? 0 : std::max(desired, quality_ - 1);
                slowAge_ = 0.0;
            }
        } else if (desired > quality_) {
            slowAge_ = 0.0;
            recoveryAge_ += sample;
            if (recoveryAge_ >= 12.0) {
                ++quality_;
                recoveryAge_ = 0.0;
            }
        } else {
            slowAge_ = 0.0;
            recoveryAge_ = 0.0;
        }

        sampleAge_ += sample;
        if (sampleAge_ >= 0.25 || frames_ == 0U) {
            displayFps_ = fps;
            displayMilliseconds_ = emaSeconds_ * 1000.0;
            sampleAge_ = 0.0;
        }
        ++frames_;
    }

    [[nodiscard]] int quality() const noexcept { return quality_; }
    [[nodiscard]] float fps() const noexcept {
        return static_cast<float>(displayFps_);
    }
    [[nodiscard]] float milliseconds() const noexcept {
        return static_cast<float>(displayMilliseconds_);
    }
    void noteBacklogDrop() noexcept { ++backlogDrops_; }
    [[nodiscard]] std::uint64_t backlogDrops() const noexcept {
        return backlogDrops_;
    }

private:
    int quality_{1};
    double emaSeconds_{1.0 / static_cast<double>(kTargetFps)};
    double displayFps_{static_cast<double>(kTargetFps)};
    double displayMilliseconds_{1000.0 / static_cast<double>(kTargetFps)};
    double sampleAge_{};
    double slowAge_{};
    double recoveryAge_{};
    std::uint64_t frames_{};
    std::uint64_t backlogDrops_{};
};

[[nodiscard]] bool containsIgnoringCase(std::string_view text,
                                        std::string_view needle) {
    if (needle.empty()) return true;
    const auto fold = [](char value) {
        if (value >= 'A' && value <= 'Z') {
            return static_cast<char>(value - 'A' + 'a');
        }
        return value;
    };
    return std::search(text.begin(), text.end(), needle.begin(), needle.end(),
                       [&](char lhs, char rhs) { return fold(lhs) == fold(rhs); })
           != text.end();
}

[[nodiscard]] bool runningOnRaspberryPi() {
    std::ifstream model("/proc/device-tree/model", std::ios::binary);
    if (!model) return false;
    std::string description((std::istreambuf_iterator<char>(model)),
                            std::istreambuf_iterator<char>());
    return containsIgnoringCase(description, "raspberry pi");
}

[[nodiscard]] bool isV3dDevice(std::string_view name) {
    return containsIgnoringCase(name, "v3d")
        || containsIgnoringCase(name, "v3dv")
        || containsIgnoringCase(name, "videocore");
}

[[nodiscard]] std::string versionString(std::uint32_t version) {
    const std::uint32_t major = version >> 22U;
    const std::uint32_t minor = (version >> 12U) & 0x3ffU;
    const std::uint32_t patch = version & 0xfffU;
    return std::to_string(major) + '.' + std::to_string(minor) + '.'
         + std::to_string(patch);
}

void placeCamera(PhysicsWorld& world, std::string_view view) {
    if (view == "pallet") {
        world.player().position = {50.0, 0.0, 22.0};
        world.player().previousPosition = world.player().position;
        world.player().yaw = 0.0;
        world.player().pitch = -0.12;
    } else if (view == "exhibits") {
        world.player().position = {29.0, 0.0, 10.5};
        world.player().previousPosition = world.player().position;
        world.player().yaw = 0.0;
        world.player().pitch = -0.22;
    } else if (view == "warehouse") {
        world.player().position = {70.0, 0.0, 20.0};
        world.player().previousPosition = world.player().position;
        world.player().yaw = 0.0;
        world.player().pitch = -0.08;
    }
}

void launchPalletStress(PhysicsWorld& world) {
    for (RigidBody& body : world.bodies()) {
        if (world.specs()[body.spec].key != "steel") continue;
        body.position = {50.0, 0.60, 25.0};
        body.previousPosition = body.position;
        body.velocity = {0.0, 0.0, 122.0};
        body.asleep = false;
        body.pristine = false;
        return;
    }
}

[[nodiscard]] constexpr std::string_view shadowStateLabel(
    bool flashlightEnabled, bool flashlightShadowsEnabled) noexcept {
    if (!flashlightShadowsEnabled) return "OFF";
    return flashlightEnabled ? "ON" : "ARMED";
}

void printVulkanConfiguration(const Renderer& renderer,
                              const PhysicsWorld& world,
                              bool gpuPhysicsEnabled,
                              bool flashlightEnabled,
                              bool flashlightShadowsEnabled,
                              bool overheadLightsEnabled,
                              bool texturesEnabled,
                              bool bumpMappingEnabled) {
    const VulkanStats graphics = renderer.vulkanStats();
    const GpuPhysicsStats compute = renderer.gpuPhysicsStats();
    std::cout << kTitle << " performance: CPU " << world.helperCount() + 1U
              << "C (main + " << world.helperCount()
              << " persistent workers); Vulkan device " << graphics.deviceName
              << "; driver " << graphics.driverName << "; API "
              << versionString(graphics.deviceApiVersion) << "; loader "
              << versionString(graphics.loaderApiVersion) << "; queue family "
              << graphics.queueFamily << "; swapchain "
              << graphics.swapchainImages << " images\n";
    std::cout << kTitle << " Vulkan profile: 1.3 "
              << (graphics.coreVulkan13 ? "core" : "KHR promotion path")
              << "; dynamic rendering "
              << (graphics.dynamicRendering ? "YES" : "NO")
              << "; synchronization2 "
              << (graphics.synchronization2 ? "YES" : "NO")
              << "; timeline semaphore "
              << (graphics.timelineSemaphore ? "YES" : "NO")
              << "; device heap " << std::fixed << std::setprecision(1)
              << static_cast<double>(graphics.deviceHeapBytes)
                    / (1024.0 * 1024.0)
              << " MiB";
    if (graphics.memoryBudget) {
        std::cout << "; live budget/usage "
                  << static_cast<double>(graphics.deviceBudgetBytes)
                        / (1024.0 * 1024.0)
                  << '/'
                  << static_cast<double>(graphics.deviceUsageBytes)
                        / (1024.0 * 1024.0)
                  << " MiB";
    }
    std::cout << '\n';
    std::cout << kTitle << " GPU secondary particle physics: "
              << (!gpuPhysicsEnabled ? "DISABLED"
                  : compute.available ? "VULKAN COMPUTE ACTIVE" : "UNAVAILABLE")
              << "; capacity " << compute.capacity << "; storage "
              << static_cast<double>(compute.storageBytes) / (1024.0 * 1024.0)
              << " MiB; zero per-frame CPU readback\n";
    std::cout << kTitle << " dynamic lighting: ceiling grid "
              << (overheadLightsEnabled ? "ON" : "OFF")
              << "; head flashlight "
              << (flashlightEnabled ? "ON" : "OFF")
              << "; flashlight shadows "
              << shadowStateLabel(flashlightEnabled,
                                  flashlightShadowsEnabled)
              << (flashlightShadowsEnabled
                      && !graphics.flashlightShadowsAvailable
                      ? " (GPU resource unavailable)" : "")
              << " (C toggles flashlight; V toggles overhead grid)\n";
    std::cout << kTitle << " surface materials: texture arrays "
              << (texturesEnabled ? "ON" : "OFF")
              << "; tangent-space bump mapping "
              << (bumpMappingEnabled && texturesEnabled ? "ON"
                  : bumpMappingEnabled ? "ARMED" : "OFF")
              << "; shared GPU-resident albedo/normal arrays; zero per-frame "
                 "texture upload or readback\n";
}

[[nodiscard]] std::optional<int> requestedQuality(
    const RuntimeOptions& options) {
    if (options.quality == "safe") return 0;
    if (options.quality == "balanced") return 1;
    if (options.quality == "ultra") return 2;
    return std::nullopt;
}

void updateWindowTitle(Platform& platform, const FrameGovernor& governor,
                       const PhysicsWorld& world,
                       const SystemTelemetry& telemetry, int quality,
                       bool flashlightEnabled,
                       bool flashlightShadowsEnabled,
                       bool overheadLightsEnabled) {
    constexpr std::array<std::string_view, 3> names{"SAFE", "BALANCED", "ULTRA"};
    std::ostringstream title;
    title << kTitle << " | " << std::fixed << std::setprecision(1)
          << governor.fps() << " FPS / " << governor.milliseconds() << " ms"
          << " | " << names[static_cast<std::size_t>(clamp(quality, 0, 2))]
          << " | CPU " << std::setprecision(0)
          << telemetry.processCpuUsagePercent << "%"
          << " | RSS " << telemetry.processResidentBytes / (1024U * 1024U)
          << " MiB | " << world.bodies().size() << " bodies"
          << " | FLASHLIGHT " << (flashlightEnabled ? "ON" : "OFF")
          << " | SHADOW "
          << shadowStateLabel(flashlightEnabled, flashlightShadowsEnabled)
          << " | OVERHEAD " << (overheadLightsEnabled ? "ON" : "OFF");
    (void)platform.setTitle(title.str());
}

int runVulkanCheck(const RuntimeOptions& options) {
    Platform platform;
    if (!platform.create(kTitle, kWindowWidth, kWindowHeight)) {
        std::cerr << "X11 window setup failed: " << platform.lastError() << '\n';
        return 1;
    }

    Renderer renderer;
    const bool maximumGpu = options.maximumThroughput
                         || options.gpuPhysics == "maximum";
    if (!renderer.initialize(platform, options.seed, maximumGpu, true, false)) {
        std::cerr << "Vulkan validation setup failed: "
                  << renderer.lastError() << '\n';
        return 1;
    }
    renderer.setGpuPhysicsEnabled(options.gpuPhysics != "off");
    renderer.setFlashlightEnabled(false);
    renderer.setFlashlightShadowsEnabled(true);
    renderer.setOverheadLightsEnabled(true);
    renderer.setMaterialTexturesEnabled(options.textures == "on");
    renderer.setNormalMappingEnabled(options.bumpMapping == "on");

    const unsigned helpers = static_cast<unsigned>(options.cpuCores - 1);
    const std::string_view backend = options.maximumThroughput
        ? std::string_view{"maximum"} : std::string_view{options.cpuBackend};
    PhysicsWorld world(options.seed, helpers, backend);
    placeCamera(world, options.view);
    if (options.stressPallet) launchPalletStress(world);

    const VulkanStats initialStats = renderer.vulkanStats();
    if (initialStats.softwareDevice) {
        std::cerr << kTitle
                  << " Vulkan smoke: FAIL (software Vulkan devices are not accepted)\n";
        return 1;
    }
    if (!initialStats.dynamicRendering || !initialStats.synchronization2) {
        std::cerr << kTitle
                  << " Vulkan smoke: FAIL (the Vulkan 1.3 feature profile "
                     "requires dynamic rendering and synchronization2)\n";
        return 1;
    }
    const bool raspberryPi = runningOnRaspberryPi();
    if (raspberryPi && !isV3dDevice(initialStats.deviceName)) {
        std::cerr << kTitle << " Vulkan smoke: FAIL (Raspberry Pi requires V3D/V3DV; got "
                  << initialStats.deviceName << ")\n";
        return 1;
    }
    if (!raspberryPi && !isV3dDevice(initialStats.deviceName)) {
        std::cout << kTitle << " Vulkan smoke: non-Pi development host; V3D identity "
                  << "check is not applicable (using " << initialStats.deviceName << ")\n";
    }

    const int requestedFrames = options.frames > 0 ? options.frames : 6;
    const int frameLimit = std::max(6, requestedFrames);
    struct LightingSmokeState {
        bool flashlight{};
        bool shadows{};
        bool overhead{};
    };
    // Prime both per-frame shadow maps in their sampled/read layout before
    // forcing each one through its first depth-write transition. Finish with
    // both explicit shadow-disabled fallback combinations.
    constexpr std::array<LightingSmokeState, 6> lightingStates{{
        {false, true, false}, {false, false, true}, {true, true, true},
        {true, true, false}, {true, false, false}, {false, false, true},
    }};
    constexpr unsigned smokeFrameContextCount = 2U;
    constexpr unsigned allSmokeFrameContexts =
        (1U << smokeFrameContextCount) - 1U;
    int rendered = 0;
    bool exercisedFlashlightOff = false;
    bool exercisedFlashlightOn = false;
    bool exercisedShadowsOff = false;
    bool exercisedShadowsOn = false;
    bool exercisedShadowsArmed = false;
    bool exercisedActiveShadows = false;
    bool exercisedOverheadOff = false;
    bool exercisedOverheadOn = false;
    unsigned inactiveShadowFrameContexts = 0U;
    unsigned transitionedShadowFrameContexts = 0U;
    std::uint64_t expectedShadowFrames = 0U;
    for (; rendered < frameLimit && !platform.closeRequested(); ++rendered) {
        if (!platform.pollEvents()) break;
        renderer.setQuality(rendered % 3);
        const LightingSmokeState lighting = lightingStates[
            static_cast<std::size_t>(rendered) % lightingStates.size()];
        renderer.setFlashlightEnabled(lighting.flashlight);
        renderer.setFlashlightShadowsEnabled(lighting.shadows);
        renderer.setOverheadLightsEnabled(lighting.overhead);
        exercisedFlashlightOff |= !lighting.flashlight;
        exercisedFlashlightOn |= lighting.flashlight;
        exercisedShadowsOff |= !lighting.shadows;
        exercisedShadowsOn |= lighting.shadows;
        exercisedShadowsArmed |= lighting.shadows && !lighting.flashlight;
        exercisedActiveShadows |= lighting.shadows && lighting.flashlight;
        exercisedOverheadOff |= !lighting.overhead;
        exercisedOverheadOn |= lighting.overhead;
        const unsigned frameContextBit =
            1U << (static_cast<unsigned>(rendered) % smokeFrameContextCount);
        if (!lighting.flashlight) {
            inactiveShadowFrameContexts |= frameContextBit;
        }
        if (lighting.shadows && lighting.flashlight) {
            if ((inactiveShadowFrameContexts & frameContextBit) != 0U) {
                transitionedShadowFrameContexts |= frameContextBit;
            }
            ++expectedShadowFrames;
        }
        world.step(kFixedDt);
        if (!renderer.render(world, 0.0F,
                             static_cast<float>(1.0 / kTargetFps))) {
            std::cerr << kTitle << " Vulkan smoke: FAIL frame " << rendered
                      << ": " << renderer.lastError() << '\n';
            return 1;
        }
    }
    renderer.waitIdle();

    const VulkanStats finalStats = renderer.vulkanStats();
    const GpuPhysicsStats finalCompute = renderer.gpuPhysicsStats();
    const bool computeExpected = options.gpuPhysics != "off";
    if (rendered < frameLimit
        || finalStats.renderedFrames < static_cast<std::uint64_t>(frameLimit)
        || (computeExpected
            && (!finalCompute.available
                || finalCompute.simulatedFrames
                    < static_cast<std::uint64_t>(frameLimit)))
        || !exercisedFlashlightOff || !exercisedFlashlightOn
        || !exercisedShadowsOff || !exercisedShadowsOn
        || !exercisedShadowsArmed || !exercisedActiveShadows
        || !exercisedOverheadOff || !exercisedOverheadOn
        || transitionedShadowFrameContexts != allSmokeFrameContexts
        || !finalStats.flashlightShadowsAvailable
        || finalStats.shadowedFrames != expectedShadowFrames
        || renderer.deviceLost() || !renderer.validationClean()) {
        std::cerr << kTitle << " Vulkan smoke: FAIL (rendered "
                  << finalStats.renderedFrames << '/' << frameLimit
                  << ", particle frames " << finalCompute.simulatedFrames << '/'
                  << (computeExpected ? frameLimit : 0)
                  << ", shadow frames " << finalStats.shadowedFrames << '/'
                  << expectedShadowFrames
                  << ", shadow resource "
                  << (finalStats.flashlightShadowsAvailable
                          ? "available" : "unavailable")
                  << ", transitioned shadow contexts "
                  << transitionedShadowFrameContexts << '/'
                  << allSmokeFrameContexts
                  << ", validation errors " << finalStats.validationErrors
                  << (renderer.deviceLost() ? ", device lost" : "") << ")\n";
        return 1;
    }

    std::cout << kTitle << " Vulkan smoke: PASS (" << frameLimit
              << " presented frames, SAFE/BALANCED/ULTRA, flashlight/shadows "
                 "OFF/ARMED/ON, overhead OFF/ON, first-use read-to-depth "
                 "transition on both "
                 "frame contexts, texture arrays "
              << (options.textures == "on" ? "ON" : "OFF")
              << ", bump mapping "
              << (options.bumpMapping == "on" && options.textures == "on"
                      ? "ON" : options.bumpMapping == "on" ? "ARMED" : "OFF")
              << ", validation clean, "
              << finalStats.deviceName << ")\n";
    return 0;
}

int runGame(const RuntimeOptions& options) {
    Platform platform;
    if (!platform.create(kTitle, kWindowWidth, kWindowHeight)) {
        std::cerr << "X11 window setup failed: " << platform.lastError() << '\n';
        return 1;
    }

    Renderer renderer;
    const bool maximumGpu = options.maximumThroughput
                         || options.gpuPhysics == "maximum";
    if (!renderer.initialize(platform, options.seed, maximumGpu, false, false)) {
        std::cerr << "Vulkan renderer setup failed: "
                  << renderer.lastError() << '\n';
        return 1;
    }
    renderer.setGpuPhysicsEnabled(options.gpuPhysics != "off");
    const bool startupFlashlightEnabled = options.flashlight == "on";
    const bool flashlightShadowsEnabled = options.shadows == "on";
    const bool startupOverheadLightsEnabled = options.overheadLights == "on";
    const bool texturesEnabled = options.textures == "on";
    const bool bumpMappingEnabled = options.bumpMapping == "on";
    const int startupMaterialViewMode = !texturesEnabled ? 2
        : bumpMappingEnabled ? 0 : 1;
    bool flashlightEnabled = startupFlashlightEnabled;
    bool overheadLightsEnabled = startupOverheadLightsEnabled;
    renderer.setFlashlightEnabled(flashlightEnabled);
    renderer.setFlashlightShadowsEnabled(flashlightShadowsEnabled);
    renderer.setOverheadLightsEnabled(overheadLightsEnabled);
    renderer.setMaterialTexturesEnabled(texturesEnabled);
    renderer.setNormalMappingEnabled(bumpMappingEnabled);

    const unsigned helpers = static_cast<unsigned>(options.cpuCores - 1);
    const std::string_view backend = options.maximumThroughput
        ? std::string_view{"maximum"} : std::string_view{options.cpuBackend};
    PhysicsWorld world(options.seed, helpers, backend);
    placeCamera(world, options.view);
    if (options.stressPallet) launchPalletStress(world);

    AudioEngine audio(!options.noAudio);
    if (options.frames == 0) (void)platform.setRelativeMouse(true);
    printVulkanConfiguration(renderer, world, options.gpuPhysics != "off",
                             flashlightEnabled, flashlightShadowsEnabled,
                             overheadLightsEnabled, texturesEnabled,
                             bumpMappingEnabled);

    FrameGovernor governor;
    const std::optional<int> qualityOverride = requestedQuality(options);
    bool running = true;
    bool hudVisible = true;
    bool helpVisible = false;
    bool simulationPaused = false;
    bool singleStepRequested = false;
    bool echoTrailsVisible = true;
    bool kineticLensEnabled = false;
    bool resonanceVisible = true;
    int materialViewMode = startupMaterialViewMode;
    bool adaptiveNormalDetail = true;
    bool gpuFloorGuard = false;
    double gpuRecoveryAge = 0.0;
    double accumulator = 0.0;
    auto previous = Clock::now();
    auto lastTitleUpdate = previous - std::chrono::seconds(1);
    const auto runStarted = previous;
    const double simulationStarted = world.simulationTime();
    std::vector<double> frameTimes;
    if (options.frames > 0) {
        frameTimes.reserve(static_cast<std::size_t>(options.frames));
    }
    const std::size_t benchmarkWarmupFrames = options.frames > 0
        ? std::min<std::size_t>(
              120U, static_cast<std::size_t>(options.frames) / 5U)
        : 0U;
    int frameCount = 0;
    std::array<double, 4> coreLoadSum{};
    double processLoadSum = 0.0;
    std::uint64_t telemetrySamples = 0;
    std::uint64_t lastTelemetrySequence = 0;
    std::uint64_t flashlightOnFrames = 0;
    std::uint64_t shadowActiveFrames = 0;
    std::uint64_t overheadLightsOnFrames = 0;
    std::uint64_t cpuPhysicsSteps = 0;
    double cpuPhysicsElapsedSeconds = 0.0;

    while (running && !platform.closeRequested()) {
        const auto now = Clock::now();
        const double rawFrameDt =
            std::chrono::duration<double>(now - previous).count();
        previous = now;
        const double frameDt = clamp(rawFrameDt, 0.0, kMaxFrameDt);
        governor.observe(frameDt > 0.0 ? frameDt
                                      : 1.0 / static_cast<double>(kTargetFps));
        const int effectiveQuality =
            qualityOverride.value_or(governor.quality());
        renderer.setQuality(effectiveQuality);

        if (maximumGpu && options.gpuPhysics != "off") {
            if (!gpuFloorGuard && governor.fps() < kMinimumFps + 10.0F) {
                gpuFloorGuard = true;
                gpuRecoveryAge = 0.0;
                renderer.setMaximumGpuPhysics(false);
            } else if (gpuFloorGuard) {
                if (governor.fps() >= 45.0F) {
                    gpuRecoveryAge += frameDt;
                    if (gpuRecoveryAge >= 3.0) {
                        gpuFloorGuard = false;
                        gpuRecoveryAge = 0.0;
                        renderer.setMaximumGpuPhysics(true);
                    }
                } else {
                    gpuRecoveryAge = 0.0;
                }
            }
        }

        if (!platform.pollEvents()) break;
        const InputState& input = platform.input();
        if (input.closeRequested || input.wentDown(Key::Escape)) running = false;
        if (input.wentDown(Key::Tab)) {
            (void)platform.setRelativeMouse(!platform.relativeMouse());
        }

        if (input.wentDown(Key::F1)) helpVisible = !helpVisible;
        if (input.wentDown(Key::F2)) hudVisible = !hudVisible;
        if (input.wentDown(Key::F3)) {
            const bool enteringPause = !simulationPaused;
            if (enteringPause && world.heldBody() >= 0) {
                // Free-look remains useful while inspecting a frozen frame.
                // Release the camera-relative spring first so rotating during
                // pause cannot inject a large correction impulse on resume.
                (void)world.pickupOrDrop();
            }
            simulationPaused = enteringPause;
            singleStepRequested = false;
            accumulator = 0.0;
        }
        if (input.wentDown(Key::F4) && simulationPaused) {
            singleStepRequested = true;
        }
        if (input.wentDown(Key::F7)) {
            echoTrailsVisible = !echoTrailsVisible;
        }
        if (input.wentDown(Key::F9)) {
            kineticLensEnabled = !kineticLensEnabled;
        }
        if (input.wentDown(Key::F10)) {
            resonanceVisible = !resonanceVisible;
        }
        if (input.wentDown(Key::F11)) {
            materialViewMode = (materialViewMode + 1) % 3;
            renderer.setMaterialTexturesEnabled(materialViewMode != 2);
            renderer.setNormalMappingEnabled(materialViewMode == 0);
        }
        if (input.wentDown(Key::F12)) {
            adaptiveNormalDetail = !adaptiveNormalDetail;
        }

        const bool resetThisFrame = input.wentDown(Key::F5);
        if (resetThisFrame) {
            world.reset();
            accumulator = 0.0;
            simulationPaused = false;
            singleStepRequested = false;
            kineticLensEnabled = false;
            resonanceVisible = true;
            materialViewMode = startupMaterialViewMode;
            adaptiveNormalDetail = true;
            renderer.setMaterialTexturesEnabled(texturesEnabled);
            renderer.setNormalMappingEnabled(bumpMappingEnabled);
            flashlightEnabled = startupFlashlightEnabled;
            renderer.setFlashlightEnabled(flashlightEnabled);
            overheadLightsEnabled = startupOverheadLightsEnabled;
            renderer.setOverheadLightsEnabled(overheadLightsEnabled);
        }
        if (input.wentDown(Key::M)) audio.toggleMute();
        if (!resetThisFrame) {
            if (input.wentDown(Key::C)) {
                flashlightEnabled = !flashlightEnabled;
                renderer.setFlashlightEnabled(flashlightEnabled);
            }
            if (input.wentDown(Key::V)) {
                overheadLightsEnabled = !overheadLightsEnabled;
                renderer.setOverheadLightsEnabled(overheadLightsEnabled);
            }
            if (!simulationPaused && input.wentDown(Key::Q)
                && world.emitEchoPulse()) {
                (void)audio.play("echo_pulse", 0.90F);
            }
            if (input.wentDown(Key::F6)) {
                world.startGalileoExperiment();
                Player& observer = world.player();
                observer.position = {10.0, 0.0, 13.0};
                observer.previousPosition = observer.position;
                observer.velocity = {};
                observer.yaw = 0.0;
                observer.pitch = 0.19;
                observer.grounded = true;
                simulationPaused = false;
                singleStepRequested = false;
                accumulator = 0.0;
            }
            if (input.wentDown(Key::F8)) {
                world.startGravityInterceptExperiment();
                Player& observer = world.player();
                observer.position = {20.5, 0.0, 13.5};
                observer.previousPosition = observer.position;
                observer.velocity = {};
                observer.yaw = 0.0;
                observer.pitch = 0.32;
                observer.grounded = true;
                simulationPaused = false;
                singleStepRequested = false;
                accumulator = 0.0;
                (void)audio.play("intercept", 0.88F);
            }
            if (!simulationPaused) {
                if (input.wentDown(Key::B)) world.togglePlayerPositionLock();
                if (input.wentDown(Key::R)) world.adjustGravity(1);
                if (input.wentDown(Key::F)) world.adjustGravity(-1);
                if (input.wentDown(Key::T)) world.adjustThrowForce(1);
                if (input.wentDown(Key::G)) world.adjustThrowForce(-1);
                if (input.wentDown(Key::Y)) world.adjustFriction(1);
                if (input.wentDown(Key::H)) world.adjustFriction(-1);
                if (input.wentDown(Key::Space) && world.jump()) {
                    audio.play("jump", 0.72F);
                }
            }
        }

        if (!resetThisFrame && platform.relativeMouse()) {
            constexpr Real sensitivity = 0.00225;
            world.player().yaw = std::fmod(
                world.player().yaw + input.mouseDeltaX * sensitivity,
                2.0 * std::numbers::pi_v<Real>);
            world.player().pitch = clamp(
                world.player().pitch - input.mouseDeltaY * sensitivity,
                -88.0 * std::numbers::pi_v<Real> / 180.0,
                88.0 * std::numbers::pi_v<Real> / 180.0);
        }
        if (!resetThisFrame && input.wentDown(MouseButton::Right)) {
            if (!platform.relativeMouse()) {
                (void)platform.setRelativeMouse(true);
            } else if (!simulationPaused) {
                (void)world.pickupOrDrop();
            }
        }
        if (!resetThisFrame && !simulationPaused
            && input.wentDown(MouseButton::Left)
            && platform.relativeMouse() && world.throwHeld()) {
            audio.play("throw", 0.70F);
        }

        const Real forward = resetThisFrame || simulationPaused ? 0.0
            : static_cast<Real>(input.down(Key::W))
                - static_cast<Real>(input.down(Key::S));
        const Real strafe = resetThisFrame || simulationPaused ? 0.0
            : static_cast<Real>(input.down(Key::D))
                - static_cast<Real>(input.down(Key::A));
        world.setMoveInput(forward, strafe);

        if (!resetThisFrame && !simulationPaused) accumulator += frameDt;
        constexpr std::array<int, 3> stepLimits{5, 8, 12};
        constexpr std::array<double, 3> physicsBudgets{0.011, 0.018, 0.028};
        constexpr std::array<int, 3> impactLimits{2, 5, 8};
        const std::size_t qualityIndex =
            static_cast<std::size_t>(clamp(effectiveQuality, 0, 2));
        int steps = 0;
        world.beginFrameEvents();
        const bool playerWasGrounded = world.player().grounded;
        const auto physicsStarted = Clock::now();
        if (!resetThisFrame && simulationPaused && singleStepRequested) {
            world.step(kFixedDt);
            accumulator = 0.0;
            singleStepRequested = false;
            steps = 1;
        }
        while (!simulationPaused && accumulator >= kFixedDt
               && steps < stepLimits[qualityIndex]) {
            world.step(kFixedDt);
            accumulator -= kFixedDt;
            ++steps;
            if (std::chrono::duration<double>(Clock::now() - physicsStarted).count()
                >= physicsBudgets[qualityIndex]) {
                break;
            }
        }
        cpuPhysicsElapsedSeconds += std::chrono::duration<double>(
            Clock::now() - physicsStarted).count();
        cpuPhysicsSteps += static_cast<std::uint64_t>(steps);
        if (!simulationPaused && accumulator >= kFixedDt) {
            accumulator = 0.0;
            governor.noteBacklogDrop();
        }

        if (!playerWasGrounded && world.player().grounded
            && world.player().landingSpeed > 0.25) {
            const float landingVolume = static_cast<float>(clamp(
                world.player().landingSpeed / 8.0, 0.18, 0.86));
            audio.play("land", landingVolume);
        }
        int impactsPlayed = 0;
        if (steps > 0) {
            for (const ImpactEvent& impact : world.impacts()) {
                if (impactsPlayed >= impactLimits[qualityIndex]) break;
                if (audio.playImpact(impact, world.player())) {
                    ++impactsPlayed;
                }
            }
        }

        (void)platform.sampleTelemetry(false);
        const SystemTelemetry& resources = platform.telemetry();
        if (resources.sampleSequence != lastTelemetrySequence) {
            lastTelemetrySequence = resources.sampleSequence;
            if (resources.sampleSequence > 1U) {
                for (std::size_t core = 0; core < coreLoadSum.size(); ++core) {
                    coreLoadSum[core] += resources.coreUsagePercent[core];
                }
                processLoadSum += resources.processCpuUsagePercent;
                ++telemetrySamples;
            }
        }
        if (now - lastTitleUpdate >= std::chrono::milliseconds(250)) {
            updateWindowTitle(platform, governor, world, resources,
                              effectiveQuality, flashlightEnabled,
                              flashlightShadowsEnabled,
                              overheadLightsEnabled);
            lastTitleUpdate = now;
        }

        const float interpolation = simulationPaused ? 1.0F
            : static_cast<float>(clamp(accumulator / kFixedDt, 0.0, 1.0));
        RendererHudState hudState{};
        hudState.visible = hudVisible;
        hudState.helpVisible = helpVisible;
        hudState.paused = simulationPaused;
        hudState.singleStep = simulationPaused && steps > 0;
        hudState.fps = static_cast<double>(governor.fps());
        hudState.quality = effectiveQuality;
        hudState.echoVisible = echoTrailsVisible;
        hudState.kineticLens = kineticLensEnabled;
        hudState.resonanceVisible = resonanceVisible;
        hudState.audioAvailable = audio.available();
        hudState.audioMuted = audio.muted();
        hudState.adaptiveNormalDetail = adaptiveNormalDetail;
        renderer.setHudState(hudState);
        const float renderDelta = simulationPaused
            ? (steps > 0 ? static_cast<float>(kFixedDt) : 0.0F)
            : static_cast<float>(frameDt);
        const bool frameRendered = renderer.render(world, interpolation,
                                                   renderDelta);
        world.endFrameEvents();
        if (!frameRendered) {
            std::cerr << "Vulkan frame failed: " << renderer.lastError() << '\n';
            return 1;
        }
        if (flashlightEnabled) ++flashlightOnFrames;
        if (flashlightEnabled && flashlightShadowsEnabled) {
            ++shadowActiveFrames;
        }
        if (overheadLightsEnabled) ++overheadLightsOnFrames;

        ++frameCount;
        if (!options.maximumThroughput) {
            const auto target = now
                + std::chrono::microseconds(1'000'000 / kTargetFps);
            std::this_thread::sleep_until(target);
        }
        if (options.frames > 0) {
            // Associate each sample with the frame just rendered. This is
            // equivalent to the next-loop delta during normal operation, but
            // it keeps the one-time warm-up timing reset out of both windows.
            frameTimes.push_back(
                std::chrono::duration<double>(Clock::now() - now).count());
        }
        if (benchmarkWarmupFrames > 0U
            && static_cast<std::size_t>(frameCount)
                == benchmarkWarmupFrames) {
            // Drain the in-flight query sets before clearing the GPU sums so
            // no warm-up submission can leak into the measured window. Reset
            // the loop clock as well: the one-time device-idle wait is setup
            // work, not a frame-time or fixed-step sample.
            renderer.resetPerformanceTimings();
            cpuPhysicsElapsedSeconds = 0.0;
            cpuPhysicsSteps = 0U;
            flashlightOnFrames = 0U;
            shadowActiveFrames = 0U;
            overheadLightsOnFrames = 0U;
            previous = Clock::now();
        }
        if (options.frames > 0 && frameCount >= options.frames) break;
    }

    renderer.waitIdle();
    if (!frameTimes.empty()) {
        const double elapsed = std::max(
            1.0e-9,
            std::chrono::duration<double>(Clock::now() - runStarted).count());
        const std::size_t warmup =
            std::min(benchmarkWarmupFrames, frameTimes.size());
        std::vector<double> steady(
            frameTimes.begin() + static_cast<std::ptrdiff_t>(warmup),
            frameTimes.end());
        if (steady.empty()) steady = frameTimes;
        std::sort(steady.begin(), steady.end());
        const auto percentile = [&](double fraction) {
            const std::size_t index = std::min(
                steady.size() - 1U,
                static_cast<std::size_t>(std::ceil(
                    fraction * static_cast<double>(steady.size())) - 1.0));
            return steady[index];
        };
        const std::size_t worstCount =
            std::max<std::size_t>(1U, (steady.size() + 99U) / 100U);
        const double worstSeconds = std::max(
            1.0e-9,
            std::accumulate(
                steady.end() - static_cast<std::ptrdiff_t>(worstCount),
                steady.end(), 0.0));
        const double onePercentLow =
            static_cast<double>(worstCount) / worstSeconds;
        const double average = std::max(
            1.0e-9,
            std::accumulate(steady.begin(), steady.end(), 0.0)
                / static_cast<double>(steady.size()));
        const auto overFloor = std::count_if(
            steady.begin(), steady.end(), [](double value) {
                return value > 1.0 / static_cast<double>(kMinimumFps);
            });

        (void)platform.sampleTelemetry(true);
        const SystemTelemetry& resources = platform.telemetry();
        const GpuPhysicsStats gpu = renderer.gpuPhysicsStats();
        const VulkanStats graphics = renderer.vulkanStats();
        const auto awakeBodies = std::count_if(
            world.bodies().begin(), world.bodies().end(),
            [](const RigidBody& body) { return !body.asleep; });
        const double sampleDivisor = static_cast<double>(
            std::max<std::uint64_t>(1U, telemetrySamples));
        const std::size_t measuredFrameCount = frameTimes.size() - warmup;
        const double cpuPhysicsMillisecondsPerFrame =
            1000.0 * cpuPhysicsElapsedSeconds
            / static_cast<double>(std::max<std::size_t>(measuredFrameCount, 1U));
        const double cpuPhysicsMillisecondsPerStep = cpuPhysicsSteps > 0U
            ? 1000.0 * cpuPhysicsElapsedSeconds
                / static_cast<double>(cpuPhysicsSteps)
            : 0.0;
        std::cout << kTitle << " benchmark: " << frameCount << " frames in "
                  << std::fixed << std::setprecision(3) << elapsed << "s = "
                  << std::setprecision(2)
                  << static_cast<double>(frameCount) / elapsed
                  << " FPS; steady " << 1.0 / average << " FPS; 1% low "
                  << onePercentLow << " FPS; p95/p99/worst "
                  << percentile(0.95) * 1000.0 << '/'
                  << percentile(0.99) * 1000.0 << '/'
                  << steady.back() * 1000.0 << " ms; over 33.3ms "
                  << overFloor << '/' << steady.size() << "; sim "
                  << (world.simulationTime() - simulationStarted) / elapsed
                  << "x; broadphase " << world.broadphaseMilliseconds()
                  << "ms; GPU particles " << gpu.activeParticles << "; awake "
                  << awakeBodies << '/' << world.bodies().size()
                  << "; Vulkan frames " << graphics.renderedFrames
                  << "; flashlight " << (flashlightEnabled ? "ON" : "OFF")
                  << " (" << flashlightOnFrames << '/' << measuredFrameCount
                  << " lit measured frames); flashlight shadows "
                  << shadowStateLabel(flashlightEnabled,
                                      flashlightShadowsEnabled)
                  << " (" << shadowActiveFrames << '/' << measuredFrameCount
                  << " active measured frames, "
                  << graphics.shadowedFrames << " GPU passes); overhead lights "
                  << (overheadLightsEnabled ? "ON" : "OFF")
                  << " (" << overheadLightsOnFrames << '/'
                  << measuredFrameCount
                  << " active measured frames); textures "
                  << (texturesEnabled ? "ON" : "OFF")
                  << "; bump mapping "
                  << (bumpMappingEnabled && texturesEnabled ? "ON"
                      : bumpMappingEnabled ? "ARMED" : "OFF")
                  << "; material view "
                  << (materialViewMode == 0 ? "FULL"
                      : materialViewMode == 1 ? "ALBEDO" : "UNIFORM")
                  << "; normal detail "
                  << (adaptiveNormalDetail ? "ADAPTIVE" : "FULL")
                  << "; CPU physics "
                  << cpuPhysicsMillisecondsPerFrame << "ms/frame, "
                  << cpuPhysicsMillisecondsPerStep << "ms/step ("
                  << cpuPhysicsSteps << " fixed steps)";
        if (graphics.gpuTimingAvailable && graphics.gpuTimedFrames > 0U) {
            std::cout << "; GPU total/particle/shadow/main "
                      << graphics.averageGpuFrameMilliseconds << '/'
                      << graphics.averageGpuComputeMilliseconds << '/'
                      << graphics.averageGpuShadowMilliseconds << '/'
                      << graphics.averageGpuGraphicsMilliseconds << " ms ("
                      << graphics.gpuTimedFrames << " timed frames)";
        } else {
            std::cout << "; GPU timing unavailable";
        }
        std::cout << "; RSS "
                  << resources.processResidentBytes / (1024U * 1024U)
                  << "MiB; cores " << std::setprecision(0)
                  << coreLoadSum[0] / sampleDivisor << "%/"
                  << coreLoadSum[1] / sampleDivisor << "%/"
                  << coreLoadSum[2] / sampleDivisor << "%/"
                  << coreLoadSum[3] / sampleDivisor << "%; process "
                  << processLoadSum / sampleDivisor << "%; RAM available "
                  << resources.availableMemoryBytes / (1024U * 1024U)
                  << "MiB; backlog drops " << governor.backlogDrops() << '\n';
    }

    if (renderer.deviceLost() || !renderer.validationClean()) {
        const VulkanStats stats = renderer.vulkanStats();
        std::cerr << kTitle << " Vulkan shutdown status: "
                  << (renderer.deviceLost() ? "device lost" : "validation failure")
                  << "; validation errors " << stats.validationErrors << '\n';
        return 1;
    }
    return 0;
}

} // namespace
} // namespace nec

int main(int argc, char** argv) {
    try {
        const nec::RuntimeOptions options = nec::parseArguments(argc, argv);
        if (!options.capturePath.empty()) {
            std::cerr << nec::kTitle
                      << ": --capture is not supported by the Vulkan foundation "
                         "renderer yet; remove --capture and try again\n";
            return 2;
        }
        if (options.check) return nec::runSelfCheck(true) ? 0 : 1;
        if (options.glCheck) return nec::runVulkanCheck(options);
        return nec::runGame(options);
    } catch (const std::exception& error) {
        std::cerr << nec::kTitle << " fatal: " << error.what() << '\n';
        return 1;
    }
}
