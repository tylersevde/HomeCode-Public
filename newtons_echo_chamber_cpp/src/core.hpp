#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <compare>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace nec {

using Clock = std::chrono::steady_clock;
using Real = double;

constexpr std::string_view kTitle =
    "Newton's Echo Chamber -- C++20 Vulkan 1.3 Pi 5 Foundation";
constexpr std::string_view kWatermark = "Made by OpenAI ChatGPT Codex 5.6 Sol Ultra";

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr int kTargetFps = 60;
constexpr int kMinimumFps = 30;
constexpr int kPhysicsHz = 120;
constexpr double kFixedDt = 1.0 / static_cast<double>(kPhysicsHz);
constexpr double kMaxFrameDt = 0.10;
constexpr int kSolverIterations = 8;

constexpr Real kRoomWidth = 100.0;
constexpr Real kRoomLength = 100.0;
constexpr Real kRoomHeight = 10.0;
constexpr Real kPlayerHeight = 2.0;
constexpr Real kPlayerEyeHeight = 1.76;
constexpr Real kPlayerRadius = 0.35;
constexpr Real kPlayerMass = 80.0;
constexpr Real kPlayerWalkSpeed = 5.0;
constexpr Real kPlayerGroundAcceleration = 24.0;
constexpr Real kPlayerAirAcceleration = 5.5;
constexpr Real kJumpHeight = 1.0;

constexpr Real kEarthGravity = 9.80665;
constexpr Real kGravityMin = 0.10;
constexpr Real kGravityMax = 50.0;
constexpr Real kGravityStep = 0.25;
constexpr Real kDefaultRoomFriction = 0.65;
constexpr Real kFrictionMin = 0.0;
constexpr Real kFrictionMax = 1.50;
constexpr Real kFrictionStep = 0.05;
constexpr Real kThrowForceMin = 1.0;
constexpr Real kThrowForceMax = 1'000'000.0;
constexpr Real kThrowStroke = 0.75;
constexpr Real kPickupReach = 6.0;
constexpr Real kHoldDistance = 1.35;
constexpr Real kAirDensity = 1.229;
constexpr Real kBroadphaseSkin = 0.006;
constexpr Real kMaximumLinearSpeed = 5'000.0;
constexpr Real kMaximumAngularSpeed = 500.0;

constexpr int kClayBrickCount = 500;
constexpr int kNoodleCount = 5;
constexpr int kGooCount = 10;
constexpr int kCeramicMarbleCount = 20;
constexpr int kStuffedAnimalCount = 5;
constexpr int kOriginalBodyCount = 551;
constexpr int kPalletBowlingBallCount = 300;
constexpr int kPalletDodgeballCount = 300;
constexpr int kPalletGooCount = 300;
constexpr int kNestedPalletCount = 25;
constexpr int kPalletMarbleCount = 3'000;
constexpr int kWarehousePalletBaseCount = 5;
constexpr int kTimberLogCount = 6;
constexpr int kExpectedBodyCount = kOriginalBodyCount
    + kPalletBowlingBallCount + kPalletDodgeballCount + kPalletGooCount
    + kNestedPalletCount + kPalletMarbleCount + kWarehousePalletBaseCount
    + kTimberLogCount;
constexpr int kMaximumActiveBricks = 4;
constexpr int kMaximumActiveBulkBodies = 128;
constexpr int kMaximumNewBulkWakes = 8;

template <typename T>
constexpr T clamp(T value, T low, T high) noexcept {
    return value < low ? low : value > high ? high : value;
}

template <typename T>
constexpr T approach(T value, T target, T amount) noexcept {
    return value < target ? std::min(target, value + amount)
                          : std::max(target, value - amount);
}

struct alignas(16) Vec3 {
    Real x{};
    Real y{};
    Real z{};

    constexpr Vec3() = default;
    constexpr Vec3(Real xValue, Real yValue, Real zValue) noexcept
        : x(xValue), y(yValue), z(zValue) {}

    [[nodiscard]] constexpr Vec3 operator+(const Vec3& rhs) const noexcept {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }
    [[nodiscard]] constexpr Vec3 operator-(const Vec3& rhs) const noexcept {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }
    [[nodiscard]] constexpr Vec3 operator-() const noexcept { return {-x, -y, -z}; }
    [[nodiscard]] constexpr Vec3 operator*(Real scalar) const noexcept {
        return {x * scalar, y * scalar, z * scalar};
    }
    [[nodiscard]] constexpr Vec3 operator/(Real scalar) const noexcept {
        return {x / scalar, y / scalar, z / scalar};
    }
    constexpr Vec3& operator+=(const Vec3& rhs) noexcept {
        x += rhs.x; y += rhs.y; z += rhs.z; return *this;
    }
    constexpr Vec3& operator-=(const Vec3& rhs) noexcept {
        x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this;
    }
    constexpr Vec3& operator*=(Real scalar) noexcept {
        x *= scalar; y *= scalar; z *= scalar; return *this;
    }
    [[nodiscard]] constexpr Real dot(const Vec3& rhs) const noexcept {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }
    [[nodiscard]] constexpr Vec3 cross(const Vec3& rhs) const noexcept {
        return {y * rhs.z - z * rhs.y,
                z * rhs.x - x * rhs.z,
                x * rhs.y - y * rhs.x};
    }
    [[nodiscard]] constexpr Real lengthSquared() const noexcept { return dot(*this); }
    [[nodiscard]] Real length() const noexcept { return std::sqrt(lengthSquared()); }
    [[nodiscard]] Vec3 normalized(Vec3 fallback = {0.0F, 0.0F, 0.0F}) const noexcept {
        const Real magnitude = length();
        return magnitude > 1.0e-12 ? *this / magnitude : fallback;
    }
    [[nodiscard]] constexpr Vec3 horizontal() const noexcept { return {x, 0.0F, z}; }
    [[nodiscard]] bool finite() const noexcept {
        return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
    }
};

[[nodiscard]] constexpr Vec3 operator*(Real scalar, const Vec3& value) noexcept {
    return value * scalar;
}

struct alignas(16) Quat {
    Real w{1.0};
    Real x{};
    Real y{};
    Real z{};

    [[nodiscard]] Quat normalized() const noexcept;
    [[nodiscard]] constexpr Quat conjugate() const noexcept { return {w, -x, -y, -z}; }
    [[nodiscard]] constexpr Quat operator*(const Quat& rhs) const noexcept {
        return {
            w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
            w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
            w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
            w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w,
        };
    }
    [[nodiscard]] Vec3 rotate(const Vec3& value) const noexcept;
    [[nodiscard]] Quat integrate(const Vec3& angularVelocity, Real dt) const noexcept;
};

enum class Shape : std::uint8_t { Sphere, Box };
enum class RenderKind : std::uint8_t {
    Default, Bat, Wheelbarrow, Balloon, Noodle, Goo, Ceramic,
    ClayBrick, Pallet, Plush, Bowling, TimberLog,
};

// Stable indices into the renderer's shared albedo and tangent-space normal
// texture arrays. Keep these values synchronized with the documented 4x4
// material atlas; zero remains a safe neutral fallback for future content.
enum class SurfaceMaterial : std::uint8_t {
    Neutral = 0,
    Concrete = 1,
    PaintedConcrete = 2,
    Rubber = 3,
    Metal = 4,
    Wood = 5,
    Latex = 6,
    Foam = 7,
    Goo = 8,
    Ceramic = 9,
    Clay = 10,
    Bowling = 11,
    Plush = 12,
};

struct BodySpec {
    std::string key;
    std::string name;
    Shape shape{Shape::Sphere};
    Real mass{};
    Real diameter{};
    Vec3 dimensions{};
    Real density{};
    Real restitution{0.3};
    Real friction{0.6};
    Real rollingResistance{0.02};
    Real dragCoefficient{0.5};
    bool thinShell{};
    std::string soundFamily{"medicine"};
    std::string rigidity;
    Vec3 color{0.8F, 0.8F, 0.8F};
    std::string category{"calibration"};
    RenderKind renderKind{RenderKind::Default};
    SurfaceMaterial surfaceMaterial{SurfaceMaterial::Neutral};
    Real buoyancyVolume{};
    Real addedMass{};
    Real adhesionStrength{};
    Real linearDamping{};

    [[nodiscard]] Real radius() const noexcept;
    [[nodiscard]] Vec3 halfExtents() const noexcept;
    [[nodiscard]] Real volume() const noexcept;
    [[nodiscard]] Real frontalArea() const noexcept;
    [[nodiscard]] Vec3 inertiaDiagonal() const noexcept;
};

struct alignas(64) RigidBody {
    std::uint16_t spec{};
    Vec3 position{};
    Vec3 velocity{};
    Quat orientation{};
    Vec3 angularVelocity{};
    Vec3 force{};
    Vec3 torque{};
    Vec3 previousPosition{};
    Quat previousOrientation{};
    Vec3 stuckNormal{};
    Vec3 stuckLocalPosition{};
    Vec3 colorOverride{-1.0F, -1.0F, -1.0F};
    Real sleepTime{};
    Real impactCooldown{};
    Real lastImpulse{};
    Real wheelAngle{};
    Real attachedPayloadMass{};
    Real cachedBoundingRadius{};
    std::int32_t stuckTo{-1};
    std::uint16_t groupIndex{};
    bool asleep{};
    bool grounded{};
    bool held{};
    bool stuckSurface{};
    bool pristine{true};
    bool massCarriedByHost{};
    std::string instanceLabel;
};

struct Player {
    Vec3 position{10.0F, 0.0F, 10.0F};
    Vec3 velocity{};
    Real yaw{};
    Real pitch{-0.30};
    Real moveForward{};
    Real moveStrafe{};
    Vec3 previousPosition{10.0F, 0.0F, 10.0F};
    Real landingSpeed{};
    bool grounded{true};

    [[nodiscard]] Vec3 eye() const noexcept;
    [[nodiscard]] Vec3 forward(bool includePitch = true) const noexcept;
    [[nodiscard]] Vec3 right() const noexcept;
};

struct ImpactEvent {
    std::string_view family;
    Vec3 position{};
    Real impulse{};
    Real speed{};
    Real mass{};
};

struct Pair {
    std::uint16_t first{};
    std::uint16_t second{};
    auto operator<=>(const Pair&) const = default;
};

class ParallelExecutor {
public:
    explicit ParallelExecutor(unsigned helperCount = 3);
    ~ParallelExecutor();
    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;

    template <typename Function>
    void parallelFor(std::size_t count, std::size_t grain, Function&& function) {
        if (count == 0) return;
        if (workers_.empty() || count <= grain) {
            function(std::size_t{0}, count);
            return;
        }
        {
            std::lock_guard lock(mutex_);
            task_ = std::forward<Function>(function);
            count_ = count;
            grain_ = std::max<std::size_t>(1, grain);
            next_.store(0, std::memory_order_relaxed);
            remaining_.store(static_cast<unsigned>(workers_.size()), std::memory_order_relaxed);
            ++generation_;
        }
        workReady_.notify_all();
        executeChunks();
        std::unique_lock lock(mutex_);
        workDone_.wait(lock, [this] {
            return remaining_.load(std::memory_order_acquire) == 0;
        });
        task_ = {};
    }

    [[nodiscard]] unsigned helperCount() const noexcept {
        return static_cast<unsigned>(workers_.size());
    }

private:
    void workerLoop(std::stop_token stop, unsigned workerIndex);
    void executeChunks();

    std::vector<std::jthread> workers_;
    std::mutex mutex_;
    std::condition_variable workReady_;
    std::condition_variable workDone_;
    std::function<void(std::size_t, std::size_t)> task_;
    std::atomic<std::size_t> next_{0};
    std::atomic<unsigned> remaining_{0};
    std::size_t count_{};
    std::size_t grain_{1};
    std::uint64_t generation_{};
    bool stopping_{};
};

struct RuntimeOptions {
    std::uint32_t seed{1337};
    int frames{};
    bool check{};
    // Kept as glCheck internally for command-line compatibility with older
    // automation. Both --vk-check and the legacy --gl-check set this flag.
    bool glCheck{};
    bool noAudio{};
    bool maximumThroughput{};
    bool stressPallet{};
    std::string view{"start"};
    std::string capturePath;
    std::string quality{"auto"};
    int cpuCores{4};
    std::string cpuBackend{"auto"};
    std::string gpuPhysics{"auto"};
    std::string flashlight{"off"};
    std::string shadows{"on"};
    std::string overheadLights{"on"};
    std::string textures{"on"};
    std::string bumpMapping{"on"};
};

[[nodiscard]] RuntimeOptions parseArguments(int argc, char** argv);
[[nodiscard]] std::vector<BodySpec> makeBodySpecs();
[[nodiscard]] bool runSelfCheck(bool verbose = true);

} // namespace nec
