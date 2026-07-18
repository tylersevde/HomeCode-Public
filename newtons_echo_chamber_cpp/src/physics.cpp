#include "physics.hpp"

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace nec {
namespace {

constexpr Real kContactEpsilon = 1.0e-12;
constexpr Real kPositionSlop = 1.0e-5;
constexpr Real kRestitutionThreshold = 0.6;
constexpr Real kWakeSpeed = 0.20;
constexpr Real kImpactSoundSpeed = 0.55;
constexpr Real kAdhesionSpeed = 0.35;
constexpr int kMaximumCcdBounces = 12;
constexpr int kMaximumNewBrickWakes = 2;
constexpr int kMaximumBulletContacts = 2;
constexpr int kMaximumBrickNeighbors = 10;
constexpr std::size_t kParallelIntegrationBodies = 64;
constexpr std::size_t kParallelBroadphaseTests = 32'768;
constexpr std::size_t kSpatialBroadphaseThreshold = 160;
constexpr Real kSpatialBroadphaseCellSize = 0.25;
constexpr std::size_t kMaximumFrameImpactEvents = 256U;
constexpr Vec3 kGalileoSteelDropPosition{9.25, 7.0, 22.0};
constexpr Vec3 kGalileoConcreteDropPosition{10.75, 7.0, 22.0};
constexpr std::size_t kGalileoMaxTrailSamples = 30U * kPhysicsHz + 1U;
constexpr Vec3 kGravityInterceptProjectilePosition{13.0, 1.2, 23.0};
constexpr Vec3 kGravityInterceptTargetPosition{28.0, 8.8, 23.0};
constexpr Real kGravityInterceptLaunchSpeed = 30.0;
constexpr double kGravityInterceptTimeout = 2.0;
constexpr std::size_t kGravityInterceptMaxTrailSamples =
    2U * kPhysicsHz + 1U;
constexpr Real kEchoPulseReach = 14.0;
constexpr Real kEchoPulseFallbackDistance = 7.0;
constexpr Real kEchoPulseRadius = 8.0;
constexpr double kEchoPulseCooldown = 0.65;
constexpr std::size_t kEchoPulseMaximumAffectedBodies = 48U;
constexpr Real kEchoPulseMaximumImpulse = 36.0;

template <typename T, std::size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;
    using is_always_equal = std::true_type;
    template <typename U> struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;
    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T*>(::operator new(count * sizeof(T),
                                              std::align_val_t{Alignment}));
    }
    void deallocate(T* pointer, std::size_t) noexcept {
        ::operator delete(pointer, std::align_val_t{Alignment});
    }
};

template <typename T, typename U, std::size_t Alignment>
constexpr bool operator==(const AlignedAllocator<T, Alignment>&,
                          const AlignedAllocator<U, Alignment>&) noexcept {
    return true;
}

using AlignedRealVector = std::vector<Real, AlignedAllocator<Real, 64>>;

constexpr std::array<std::string_view, 5> kBulkCategories{
    "bulk_bowling", "bulk_dodge", "bulk_goo", "bulk_marble", "bulk_pallet",
};
constexpr std::array<int, 5> kBulkFamilyCaps{32, 32, 32, 64, 16};

[[nodiscard]] int bulkFamilyIndex(const BodySpec& spec) noexcept {
    for (std::size_t index = 0; index < kBulkCategories.size(); ++index) {
        if (spec.category == kBulkCategories[index]) return static_cast<int>(index);
    }
    return -1;
}

[[nodiscard]] bool isBulkStock(const BodySpec& spec) noexcept {
    return bulkFamilyIndex(spec) >= 0;
}

[[nodiscard]] bool isCourseStock(const BodySpec& spec) noexcept {
    return spec.key == "clay_brick" || spec.key == "wood_pallet"
        || spec.category == "bulk_pallet";
}

[[nodiscard]] std::string numberedLabel(std::string_view prefix, int number,
                                        int total, int width = 0) {
    std::ostringstream stream;
    stream << prefix << ' ';
    if (width > 0) stream << std::setfill('0') << std::setw(width);
    stream << number << '/' << total;
    return stream.str();
}

[[nodiscard]] Vec3 velocityAt(const RigidBody& body, const Vec3& point) noexcept {
    return body.velocity + body.angularVelocity.cross(point - body.position);
}

[[nodiscard]] Vec3 inverseInertiaWorld(const BodySpec& spec,
                                       const RigidBody& body,
                                       const Vec3& value) noexcept {
    const Vec3 local = body.orientation.conjugate().rotate(value);
    const Vec3 inertia = spec.inertiaDiagonal();
    const Vec3 transformed{
        local.x / std::max(inertia.x, 1.0e-9),
        local.y / std::max(inertia.y, 1.0e-9),
        local.z / std::max(inertia.z, 1.0e-9),
    };
    return body.orientation.rotate(transformed);
}

void applyImpulseRaw(const BodySpec& spec, RigidBody& body, Real inverseMass,
                     const Vec3& impulse, const Vec3& point) noexcept {
    body.velocity += impulse * inverseMass;
    const Vec3 angularImpulse = (point - body.position).cross(impulse);
    body.angularVelocity += inverseInertiaWorld(spec, body, angularImpulse);
}

[[nodiscard]] std::array<Vec3, 3> boxAxes(const RigidBody& body) noexcept {
    return {
        body.orientation.rotate({1.0, 0.0, 0.0}).normalized({1.0, 0.0, 0.0}),
        body.orientation.rotate({0.0, 1.0, 0.0}).normalized({0.0, 1.0, 0.0}),
        body.orientation.rotate({0.0, 0.0, 1.0}).normalized({0.0, 0.0, 1.0}),
    };
}

[[nodiscard]] Real projectedRadius(const std::array<Vec3, 3>& axes,
                                   const Vec3& half,
                                   const Vec3& axis) noexcept {
    return half.x * std::abs(axis.dot(axes[0]))
         + half.y * std::abs(axis.dot(axes[1]))
         + half.z * std::abs(axis.dot(axes[2]));
}

[[nodiscard]] Real closestSweepFraction(const RigidBody& a,
                                        const RigidBody& b) noexcept {
    const Vec3 start = b.previousPosition - a.previousPosition;
    const Vec3 motion = (b.position - b.previousPosition)
                      - (a.position - a.previousPosition);
    const Real motion2 = motion.lengthSquared();
    return motion2 > 1.0e-14 ? clamp(-start.dot(motion) / motion2, 0.0, 1.0)
                             : 0.0;
}

[[nodiscard]] std::uint32_t pairCode(const Pair& pair) noexcept {
    return (static_cast<std::uint32_t>(pair.first) << 16U)
         | static_cast<std::uint32_t>(pair.second);
}

[[nodiscard]] Real roundedTo(Real value, Real scale) noexcept {
    return std::nearbyint(value * scale) / scale;
}

[[nodiscard]] BodySpec makeSpec(
    std::string key, std::string name, Shape shape, Real mass,
    Real diameter, Vec3 dimensions, Real density, Real restitution,
    Real friction, Real rollingResistance, Real dragCoefficient,
    bool thinShell, std::string soundFamily, std::string rigidity,
    Vec3 color, std::string category = "calibration",
    RenderKind renderKind = RenderKind::Default, Real buoyancyVolume = 0.0,
    Real addedMass = 0.0, Real adhesionStrength = 0.0,
    Real linearDamping = 0.0) {
    BodySpec spec;
    spec.key = std::move(key);
    spec.name = std::move(name);
    spec.shape = shape;
    spec.mass = mass;
    spec.diameter = diameter;
    spec.dimensions = dimensions;
    spec.density = density;
    spec.restitution = restitution;
    spec.friction = friction;
    spec.rollingResistance = rollingResistance;
    spec.dragCoefficient = dragCoefficient;
    spec.thinShell = thinShell;
    spec.soundFamily = std::move(soundFamily);
    spec.rigidity = std::move(rigidity);
    spec.color = color;
    spec.category = std::move(category);
    spec.renderKind = renderKind;
    spec.buoyancyVolume = buoyancyVolume;
    spec.addedMass = addedMass;
    spec.adhesionStrength = adhesionStrength;
    spec.linearDamping = linearDamping;
    return spec;
}

} // namespace

Quat Quat::normalized() const noexcept {
    const Real magnitude = std::sqrt(w * w + x * x + y * y + z * z);
    return magnitude > 1.0e-12
        ? Quat{w / magnitude, x / magnitude, y / magnitude, z / magnitude}
        : Quat{};
}

Vec3 Quat::rotate(const Vec3& value) const noexcept {
    const Quat result = (*this) * Quat{0.0, value.x, value.y, value.z}
                      * conjugate();
    return {result.x, result.y, result.z};
}

Quat Quat::integrate(const Vec3& angularVelocity, Real dt) const noexcept {
    const Quat derivative = Quat{0.0, angularVelocity.x,
                                 angularVelocity.y, angularVelocity.z} * (*this);
    return Quat{
        w + 0.5 * derivative.w * dt,
        x + 0.5 * derivative.x * dt,
        y + 0.5 * derivative.y * dt,
        z + 0.5 * derivative.z * dt,
    }.normalized();
}

Real BodySpec::radius() const noexcept {
    return shape == Shape::Sphere
        ? diameter * 0.5
        : 0.5 * std::max({dimensions.x, dimensions.y, dimensions.z});
}

Vec3 BodySpec::halfExtents() const noexcept { return dimensions * 0.5; }

Real BodySpec::volume() const noexcept {
    if (shape == Shape::Sphere) {
        return 4.0 * std::numbers::pi_v<Real> * std::pow(radius(), 3.0) / 3.0;
    }
    return dimensions.x * dimensions.y * dimensions.z;
}

Real BodySpec::frontalArea() const noexcept {
    return shape == Shape::Sphere
        ? std::numbers::pi_v<Real> * radius() * radius()
        : dimensions.x * dimensions.y;
}

Vec3 BodySpec::inertiaDiagonal() const noexcept {
    const Real inertialMass = mass + addedMass;
    if (shape == Shape::Sphere) {
        const Real factor = thinShell ? 2.0 / 3.0 : 2.0 / 5.0;
        const Real inertia = factor * inertialMass * radius() * radius();
        return {inertia, inertia, inertia};
    }
    const Real x2 = dimensions.x * dimensions.x;
    const Real y2 = dimensions.y * dimensions.y;
    const Real z2 = dimensions.z * dimensions.z;
    return {
        inertialMass * (y2 + z2) / 12.0,
        inertialMass * (x2 + z2) / 12.0,
        inertialMass * (x2 + y2) / 12.0,
    };
}

Vec3 Player::eye() const noexcept {
    return position + Vec3{0.0, kPlayerEyeHeight, 0.0};
}

Vec3 Player::forward(bool includePitch) const noexcept {
    const Real usedPitch = includePitch ? pitch : 0.0;
    const Real cosine = std::cos(usedPitch);
    return {-std::sin(yaw) * cosine, std::sin(usedPitch),
            std::cos(yaw) * cosine};
}

Vec3 Player::right() const noexcept {
    return {-std::cos(yaw), 0.0, -std::sin(yaw)};
}

ParallelExecutor::ParallelExecutor(unsigned helperCount) {
    const unsigned hardware = std::max(1U, std::thread::hardware_concurrency());
    helperCount = std::min(helperCount, hardware - 1U);
    workers_.reserve(helperCount);
    for (unsigned index = 0; index < helperCount; ++index) {
        workers_.emplace_back([this, index](std::stop_token stop) {
            workerLoop(stop, index);
        });
    }
}

ParallelExecutor::~ParallelExecutor() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        ++generation_;
    }
    for (auto& worker : workers_) worker.request_stop();
    workReady_.notify_all();
    // Join while the mutex/condition variables still exist; member destruction
    // would otherwise destroy those synchronization objects before workers_.
    workers_.clear();
}

void ParallelExecutor::executeChunks() {
    for (;;) {
        const std::size_t begin = next_.fetch_add(grain_, std::memory_order_relaxed);
        if (begin >= count_) break;
        task_(begin, std::min(count_, begin + grain_));
    }
}

void ParallelExecutor::workerLoop(std::stop_token stop, unsigned workerIndex) {
#if defined(__linux__)
    cpu_set_t available;
    CPU_ZERO(&available);
    if (pthread_getaffinity_np(pthread_self(), sizeof(available), &available) == 0) {
        std::vector<int> cores;
        for (int core = 0; core < CPU_SETSIZE; ++core) {
            if (CPU_ISSET(core, &available)) cores.push_back(core);
        }
        if (!cores.empty()) {
            const std::size_t slot = std::min<std::size_t>(workerIndex + 1,
                                                           cores.size() - 1);
            cpu_set_t selected;
            CPU_ZERO(&selected);
            CPU_SET(cores[slot], &selected);
            (void)pthread_setaffinity_np(pthread_self(), sizeof(selected), &selected);
        }
    }
#else
    (void)workerIndex;
#endif
    std::uint64_t observed = 0;
    for (;;) {
        std::unique_lock lock(mutex_);
        workReady_.wait(lock, [&] {
            return stopping_ || stop.stop_requested() || generation_ != observed;
        });
        if (stopping_ || stop.stop_requested()) return;
        observed = generation_;
        lock.unlock();
        executeChunks();
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Serialize the completion notification with the waiter's
            // predicate-check-to-sleep transition.  remaining_ is atomic, but
            // notifying without this mutex still permits the final signal to
            // land after parallelFor() observes a non-zero count and before
            // its condition-variable wait actually blocks.  That lost wake
            // leaves the main thread asleep with remaining_ == 0 forever.
            std::lock_guard completionLock(mutex_);
            workDone_.notify_one();
        }
    }
}

std::vector<BodySpec> makeBodySpecs() {
    const Real basketballDiameter = 0.760 / std::numbers::pi_v<Real>;
    const Real basketballRadius = basketballDiameter * 0.5;
    const Real basketballVolume = 4.0 * std::numbers::pi_v<Real>
                                * std::pow(basketballRadius, 3.0) / 3.0;
    const Vec3 rubberBrickDimensions{0.194, 0.092, 0.057};
    std::vector<BodySpec> specs;
    specs.reserve(31);

    specs.push_back(makeSpec(
        "dodge", "Inflated Rubber Dodgeball", Shape::Sphere, 0.310,
        0.660 / std::numbers::pi_v<Real>, {},
        0.310 / (4.0 * std::numbers::pi_v<Real>
                 * std::pow(0.330 / std::numbers::pi_v<Real>, 3.0) / 3.0),
        0.72, 0.80, 0.025, 0.50, true, "dodge",
        "inflated compliant shell", {0.96, 0.22, 0.18}));
    specs.push_back(makeSpec(
        "medicine_1", "Medicine Ball 1 kg", Shape::Sphere, 1.0, 0.190, {},
        1.0 / (4.0 * std::numbers::pi_v<Real> * std::pow(0.095, 3.0) / 3.0),
        0.40, 0.75, 0.040, 0.50, false, "medicine",
        "compliant filled rubber", {0.15, 0.54, 0.94}));
    specs.push_back(makeSpec(
        "medicine_3", "Medicine Ball 3 kg", Shape::Sphere, 3.0, 0.230, {},
        3.0 / (4.0 * std::numbers::pi_v<Real> * std::pow(0.115, 3.0) / 3.0),
        0.32, 0.75, 0.045, 0.50, false, "medicine",
        "semicompliant filled rubber", {0.20, 0.75, 0.42}));
    specs.push_back(makeSpec(
        "medicine_10", "Medicine Ball 10 kg", Shape::Sphere, 10.0, 0.280, {},
        10.0 / (4.0 * std::numbers::pi_v<Real> * std::pow(0.140, 3.0) / 3.0),
        0.22, 0.75, 0.060, 0.50, false, "medicine",
        "dense damped fill", {0.69, 0.30, 0.86}));
    specs.push_back(makeSpec(
        "steel", "Solid Smooth-Steel Ball", Shape::Sphere,
        basketballVolume * 7'850.0, basketballDiameter, {}, 7'850.0,
        0.55, 0.35, 0.003, 0.47, false, "steel",
        "rigid steel (~200 GPa)", {0.70, 0.76, 0.82}));
    specs.push_back(makeSpec(
        "concrete", "Grainy Concrete Ball", Shape::Sphere,
        basketballVolume * 2'400.0, basketballDiameter, {}, 2'400.0,
        0.18, 0.65, 0.020, 0.50, false, "concrete",
        "rigid brittle concrete", {0.48, 0.45, 0.40}));
    specs.push_back(makeSpec(
        "rubber_brick", "Solid Natural-Rubber Brick", Shape::Box,
        rubberBrickDimensions.x * rubberBrickDimensions.y
            * rubberBrickDimensions.z * 920.0,
        0.0, rubberBrickDimensions, 920.0, 0.30, 0.85, 0.0, 1.05,
        false, "rubber_brick", "solid compliant rubber", {0.90, 0.48, 0.09}));
    specs.push_back(makeSpec(
        "wood_bat", "Ash Wooden Baseball Bat", Shape::Box, 0.8788, 0.0,
        {0.8636, 0.0663, 0.0663}, 678.0, 0.32, 0.55, 0.010, 1.20,
        false, "wood_bat", "hard tapered ash", {0.73, 0.46, 0.20},
        "bat", RenderKind::Bat));
    specs.push_back(makeSpec(
        "wheelbarrow", "Single-Wheel Steel Wheelbarrow", Shape::Box, 17.28,
        0.0, {0.648, 0.686, 1.492}, 0.0, 0.12, 0.62, 0.018, 1.15,
        false, "wheelbarrow", "steel tray, hardwood handles, pneumatic tire",
        {0.22, 0.48, 0.32}, "wheelbarrow", RenderKind::Wheelbarrow));
    specs.push_back(makeSpec(
        "helium_balloon", "Three-Foot Helium Latex Balloon", Shape::Sphere,
        0.183, 0.914, {}, 0.431, 0.55, 0.80, 0.0, 0.50, true,
        "balloon", "compliant inflated latex", {0.96, 0.35, 0.78},
        "balloon", RenderKind::Balloon, 0.4248, 0.261, 0.0, 0.04));
    specs.push_back(makeSpec(
        "foam_noodle", "Closed-Cell PE Foam Noodle", Shape::Box, 0.1134,
        0.0, {1.1938, 0.0635, 0.0635}, 30.0, 0.25, 0.65, 0.060, 1.20,
        false, "foam_noodle", "flexible PE foam (damped rigid proxy)",
        {0.22, 0.88, 0.92}, "noodles", RenderKind::Noodle,
        0.0, 0.0, 0.0, 0.11));
    specs.push_back(makeSpec(
        "sticky_goo", "Sticky PVA Goo Blob", Shape::Sphere, 0.248, 0.0767,
        {}, 1'050.0, 0.02, 1.10, 0.20, 0.60, false, "goo",
        "viscoelastic adhesive hydrogel", {0.22, 0.96, 0.36},
        "goo", RenderKind::Goo, 0.0, 0.0, 25.0, 1.80));
    specs.push_back(makeSpec(
        "ceramic_marble", "Oversized Alumina Ceramic Marble", Shape::Sphere,
        0.2356, 0.050, {}, 3'600.0, 0.55, 0.28, 0.003, 0.47, false,
        "ceramic", "hard 92% alumina ceramic", {0.94, 0.94, 0.89},
        "marbles", RenderKind::Ceramic));
    specs.push_back(makeSpec(
        "clay_brick", "Modular Fired-Clay Brick", Shape::Box, 1.90, 0.0,
        {0.194, 0.057, 0.092}, 1'868.0, 0.12, 0.68, 0.0, 1.05,
        false, "clay_brick", "rigid brittle fired clay", {0.57, 0.19, 0.09},
        "bricks", RenderKind::ClayBrick));
    specs.push_back(makeSpec(
        "wood_pallet", "EPAL 1 Wooden Pallet", Shape::Box, 25.0, 0.0,
        {1.200, 0.144, 0.800}, 181.0, 0.10, 0.62, 0.0, 1.20,
        false, "pallet", "spruce/pine pallet rated 1,500 kg dynamic / 4,000 kg static",
        {0.52, 0.33, 0.16}, "pallet", RenderKind::Pallet));

    constexpr Real bowlingDiameter = 0.2183;
    constexpr Real bowlingMass = 6.80388555; // representative 15 lb adult ball
    const Real bowlingRadius = bowlingDiameter * 0.5;
    const Real bowlingVolume = 4.0 * std::numbers::pi_v<Real>
                             * bowlingRadius * bowlingRadius * bowlingRadius / 3.0;
    specs.push_back(makeSpec(
        "bowling_ball", "15 lb Urethane Bowling Ball", Shape::Sphere,
        bowlingMass, bowlingDiameter, {}, bowlingMass / bowlingVolume,
        0.18, 0.24, 0.012, 0.47, false, "bowling",
        "rigid polyester/urethane bowling shell", {0.16, 0.32, 0.78},
        "bulk_bowling", RenderKind::Bowling));

    constexpr Real timberLength = 2.40;
    constexpr Real timberDiameter = 0.30;
    constexpr Real timberDensity = 530.0;
    const Real timberMass = std::numbers::pi_v<Real>
                          * std::pow(timberDiameter * 0.5, 2.0)
                          * timberLength * timberDensity;
    specs.push_back(makeSpec(
        "timber_log", "Seasoned Douglas-Fir Timber Log", Shape::Box,
        timberMass, 0.0, {timberLength, timberDiameter, timberDiameter},
        timberDensity, 0.12, 0.68, 0.040, 1.05, false, "timber",
        "seasoned Douglas-fir cylinder (box collision proxy)", {0.43, 0.25, 0.10},
        "timber", RenderKind::TimberLog));

    const auto appendVariant = [&](std::string_view sourceKey, std::string key,
                                   std::string name, std::string category) {
        const auto source = std::find_if(specs.begin(), specs.end(),
            [&](const BodySpec& spec) { return spec.key == sourceKey; });
        if (source == specs.end()) throw std::logic_error("bulk source spec missing");
        BodySpec variant = *source;
        variant.key = std::move(key);
        variant.name = std::move(name);
        variant.category = std::move(category);
        specs.push_back(std::move(variant));
    };
    appendVariant("dodge", "pallet_dodge", "Palletized Rubber Dodgeball",
                  "bulk_dodge");
    appendVariant("sticky_goo", "pallet_goo", "Palletized Sticky Goo Glob",
                  "bulk_goo");
    appendVariant("ceramic_marble", "pallet_marble",
                  "Palletized Alumina Ceramic Marble", "bulk_marble");
    appendVariant("wood_pallet", "bulk_pallet", "Warehouse EPAL 1 Pallet",
                  "bulk_pallet");

    struct PlushData {
        const char* key;
        const char* label;
        Real diameter;
        Real mass;
        Vec3 color;
    };
    constexpr std::array plushData{
        PlushData{"bear", "Bear", 0.381, 0.2835, {0.72, 0.54, 0.36}},
        PlushData{"rabbit", "Rabbit", 0.410, 0.310, {0.91, 0.86, 0.78}},
        PlushData{"fox", "Fox", 0.390, 0.295, {0.92, 0.39, 0.16}},
        PlushData{"penguin", "Penguin", 0.360, 0.270, {0.18, 0.22, 0.29}},
        PlushData{"dinosaur", "Dinosaur", 0.460, 0.345, {0.28, 0.67, 0.36}},
        PlushData{"octopus", "Octopus", 0.350, 0.255, {0.68, 0.35, 0.86}},
        PlushData{"axolotl", "Axolotl", 0.400, 0.285, {0.98, 0.55, 0.66}},
        PlushData{"elephant", "Elephant", 0.430, 0.335, {0.55, 0.61, 0.68}},
        PlushData{"raccoon", "Raccoon", 0.370, 0.280, {0.42, 0.45, 0.49}},
        PlushData{"sloth", "Sloth", 0.420, 0.320, {0.48, 0.34, 0.25}},
    };
    for (const auto& data : plushData) {
        const Real radius = data.diameter * 0.5;
        const Real volume = 4.0 * std::numbers::pi_v<Real>
                          * radius * radius * radius / 3.0;
        specs.push_back(makeSpec(
            std::string("plush_") + data.key,
            std::string("Stuffed ") + data.label,
            Shape::Sphere, data.mass, data.diameter, {}, data.mass / volume,
            0.08, 0.75, 0.12, 1.10, false, "plush",
            "soft polyester shell and hollow-fiber fill", data.color,
            "plush", RenderKind::Plush, 0.0, 0.0, 0.0, 0.42));
    }

    // Assign every current rigid-body proxy a stable texture-array layer once
    // at catalog construction. Rendering can then pack the layer without doing
    // per-visible-body string comparisons each frame.
    for (BodySpec& spec : specs) {
        switch (spec.renderKind) {
        case RenderKind::Bat:
        case RenderKind::Pallet:
        case RenderKind::TimberLog:
            spec.surfaceMaterial = SurfaceMaterial::Wood;
            break;
        case RenderKind::Wheelbarrow:
            spec.surfaceMaterial = SurfaceMaterial::Metal;
            break;
        case RenderKind::Balloon:
            spec.surfaceMaterial = SurfaceMaterial::Latex;
            break;
        case RenderKind::Noodle:
            spec.surfaceMaterial = SurfaceMaterial::Foam;
            break;
        case RenderKind::Goo:
            spec.surfaceMaterial = SurfaceMaterial::Goo;
            break;
        case RenderKind::Ceramic:
            spec.surfaceMaterial = SurfaceMaterial::Ceramic;
            break;
        case RenderKind::ClayBrick:
            spec.surfaceMaterial = SurfaceMaterial::Clay;
            break;
        case RenderKind::Plush:
            spec.surfaceMaterial = SurfaceMaterial::Plush;
            break;
        case RenderKind::Bowling:
            spec.surfaceMaterial = SurfaceMaterial::Bowling;
            break;
        case RenderKind::Default:
            spec.surfaceMaterial = spec.key == "concrete"
                ? SurfaceMaterial::Concrete
                : spec.key == "steel"
                    ? SurfaceMaterial::Metal
                    : SurfaceMaterial::Rubber;
            break;
        }
    }
    return specs;
}

PhysicsWorld::PhysicsWorld(std::uint32_t seed, unsigned helpers,
                           std::string_view cpuBackend)
    : seed_(seed), rng_(seed), specs_(makeBodySpecs()),
      executor_(std::min(helpers, 3U)), cpuBackend_(cpuBackend) {
    if (cpuBackend_ != "scalar" && cpuBackend_ != "auto"
        && cpuBackend_ != "maximum") {
        throw std::invalid_argument("unknown CPU backend: " + cpuBackend_);
    }
    specByKey_.reserve(specs_.size());
    for (std::size_t index = 0; index < specs_.size(); ++index) {
        specByKey_.emplace(specs_[index].key, static_cast<std::uint16_t>(index));
    }
    bodies_.reserve(kExpectedBodyCount);
    activeIndices_.reserve(kExpectedBodyCount);
    broadphasePairs_.reserve(8192);
    contacts_.reserve(1024);
    impacts_.reserve(256);
    pendingWakes_.reserve(256);
    galileoCurrentTrail_.reserve(kGalileoMaxTrailSamples);
    galileoPreviousTrail_.reserve(kGalileoMaxTrailSamples);
    gravityInterceptCurrentTrail_.reserve(kGravityInterceptMaxTrailSamples);
    gravityInterceptPreviousTrail_.reserve(kGravityInterceptMaxTrailSamples);
    lastEchoPulseBodies_.reserve(kEchoPulseMaximumAffectedBodies);
    reset();
}

std::uint16_t PhysicsWorld::specIndex(std::string_view key) const {
    const auto iterator = specByKey_.find(std::string(key));
    if (iterator == specByKey_.end()) {
        throw std::out_of_range("unknown body spec: " + std::string(key));
    }
    return iterator->second;
}

void PhysicsWorld::spawn(std::uint16_t specIndexValue, const Vec3& position,
                         std::string label, int groupIndex, bool asleep,
                         bool grounded, Quat orientation, Vec3 color) {
    if (specIndexValue >= specs_.size()) {
        throw std::out_of_range("body spec index outside catalog");
    }
    RigidBody body;
    body.spec = specIndexValue;
    body.position = position;
    body.previousPosition = position;
    body.orientation = orientation.normalized();
    body.previousOrientation = body.orientation;
    body.instanceLabel = std::move(label);
    body.groupIndex = static_cast<std::uint16_t>(std::max(0, groupIndex));
    body.asleep = asleep;
    body.grounded = grounded;
    body.colorOverride = color;
    const BodySpec& spec = specs_[specIndexValue];
    if (spec.shape == Shape::Sphere) {
        body.cachedBoundingRadius = spec.radius();
    } else {
        body.cachedBoundingRadius = spec.halfExtents().length();
    }
    bodies_.push_back(std::move(body));
}

void PhysicsWorld::reset() {
    rng_.seed(seed_);
    gravity_ = kEarthGravity;
    roomFriction_ = kDefaultRoomFriction;
    throwForce_ = kThrowForceMin;
    player_ = Player{};
    playerPositionLocked_ = false;
    lockedPlayerPosition_ = player_.position;
    wheelbarrowIndex_ = -1;
    bodies_.clear();
    bodies_.reserve(kExpectedBodyCount);
    impacts_.clear();
    collectingFrameEvents_ = false;
    pendingWakes_.clear();
    contacts_.clear();
    broadphasePairs_.clear();
    activeIndices_.clear();

    constexpr std::array<std::string_view, 7> originals{
        "dodge", "medicine_1", "medicine_3", "medicine_10",
        "steel", "concrete", "rubber_brick",
    };
    for (std::size_t index = 0; index < originals.size(); ++index) {
        const std::uint16_t specId = specIndex(originals[index]);
        const BodySpec& spec = specs_[specId];
        const Real y = spec.shape == Shape::Sphere
            ? spec.radius() : spec.dimensions.y * 0.5;
        spawn(specId, {7.0 + static_cast<Real>(index), y, 15.0}, {},
              0, false, true);
    }

    const auto batId = specIndex("wood_bat");
    spawn(batId, {15.0, specs_[batId].dimensions.y * 0.5, 15.5},
          {}, 0, false, true);

    const auto wheelbarrowId = specIndex("wheelbarrow");
    spawn(wheelbarrowId,
          {18.5, specs_[wheelbarrowId].dimensions.y * 0.5, 18.0},
          {}, 0, true, true);
    wheelbarrowIndex_ = static_cast<int>(bodies_.size()) - 1;

    const auto balloonId = specIndex("helium_balloon");
    spawn(balloonId, {21.5, 2.7, 17.0});

    constexpr std::array<Vec3, 5> noodleColors{
        Vec3{0.18, 0.90, 0.96}, Vec3{0.98, 0.30, 0.35},
        Vec3{0.98, 0.82, 0.18}, Vec3{0.44, 0.95, 0.32},
        Vec3{0.74, 0.38, 0.96},
    };
    const auto noodleId = specIndex("foam_noodle");
    const Real noodleY = specs_[noodleId].dimensions.y * 0.5;
    for (int index = 0; index < kNoodleCount; ++index) {
        spawn(noodleId, {24.5, noodleY, 15.0 + index * 0.24},
              numberedLabel("Foam Noodle", index + 1, kNoodleCount),
              index + 1, true, true, {}, noodleColors[static_cast<std::size_t>(index)]);
    }

    constexpr std::array<Vec3, 3> gooColors{
        Vec3{0.20, 0.98, 0.35}, Vec3{0.20, 0.78, 1.0},
        Vec3{0.92, 0.30, 0.98},
    };
    const auto gooId = specIndex("sticky_goo");
    for (int index = 0; index < kGooCount; ++index) {
        spawn(gooId,
              {27.0 + (index % 5) * 0.14, specs_[gooId].radius(),
               15.0 + (index / 5) * 0.16},
              numberedLabel("Sticky Goo", index + 1, kGooCount),
              index + 1, true, true, {},
              gooColors[static_cast<std::size_t>(index) % gooColors.size()]);
    }

    constexpr std::array<Vec3, 3> marbleColors{
        Vec3{0.96, 0.95, 0.89}, Vec3{0.54, 0.78, 0.98},
        Vec3{0.98, 0.70, 0.35},
    };
    const auto marbleId = specIndex("ceramic_marble");
    for (int index = 0; index < kCeramicMarbleCount; ++index) {
        spawn(marbleId,
              {30.0 + (index % 5) * 0.075, specs_[marbleId].radius(),
               15.0 + (index / 5) * 0.075},
              numberedLabel("Ceramic Marble", index + 1,
                            kCeramicMarbleCount),
              index + 1, true, true, {},
              marbleColors[static_cast<std::size_t>(index) % marbleColors.size()]);
    }

    const auto palletId = specIndex("wood_pallet");
    const Vec3 palletCenter{50.0, specs_[palletId].dimensions.y * 0.5, 28.0};
    spawn(palletId, palletCenter, {}, 0, true, true);

    const auto brickId = specIndex("clay_brick");
    const BodySpec& brick = specs_[brickId];
    int brickNumber = 0;
    const Real palletTop = specs_[palletId].dimensions.y;
    const Real layerPitch = brick.dimensions.y + 0.001;
    const Quat quarterYaw{std::cos(std::numbers::pi_v<Real> / 4.0), 0.0,
                          std::sin(std::numbers::pi_v<Real> / 4.0), 0.0};
    for (int layer = 0; layer < 10; ++layer) {
        const bool odd = (layer & 1) != 0;
        const int across = odd ? 12 : 6;
        const int deep = odd ? 4 : 8;
        const Real xPitch = (odd ? brick.dimensions.z : brick.dimensions.x) + 0.001;
        const Real zPitch = (odd ? brick.dimensions.x : brick.dimensions.z) + 0.001;
        const Quat orientation = odd ? quarterYaw : Quat{};
        for (int row = 0; row < deep; ++row) {
            for (int column = 0; column < across; ++column) {
                ++brickNumber;
                const Vec3 color{
                    0.50 + 0.018 * static_cast<Real>((brickNumber * 7) % 5),
                    0.15, 0.065,
                };
                spawn(brickId,
                      {palletCenter.x
                           + (column - (across - 1) * 0.5) * xPitch,
                       palletTop + brick.dimensions.y * 0.5
                           + layer * layerPitch,
                       palletCenter.z
                           + (row - (deep - 1) * 0.5) * zPitch},
                      numberedLabel("Clay Brick", brickNumber,
                                    kClayBrickCount, 3),
                      brickNumber, true, true, orientation, color);
            }
        }
    }
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 5; ++column) {
            ++brickNumber;
            const Vec3 color{
                0.50 + 0.018 * static_cast<Real>((brickNumber * 7) % 5),
                0.15, 0.065,
            };
            spawn(brickId,
                  {palletCenter.x + (column - 2.0) * (brick.dimensions.x + 0.001),
                   palletTop + brick.dimensions.y * 0.5 + 10.0 * layerPitch,
                   palletCenter.z + (row - 1.5) * (brick.dimensions.z + 0.001)},
                  numberedLabel("Clay Brick", brickNumber,
                                kClayBrickCount, 3),
                  brickNumber, true, true, {}, color);
        }
    }

    // Five isolated warehouse loads. Every item starts awake and pristine;
    // the sparse spatial broadphase keeps the full 4,487-body opening stable.
    constexpr std::array<Vec3, kWarehousePalletBaseCount> warehouseCenters{
        Vec3{54.0, 0.0, 44.0}, Vec3{62.0, 0.0, 44.0},
        Vec3{70.0, 0.0, 44.0}, Vec3{78.0, 0.0, 44.0},
        Vec3{86.0, 0.0, 44.0},
    };
    constexpr std::array<std::string_view, kWarehousePalletBaseCount> baseLabels{
        "Bowling Ball Pallet Base", "Dodgeball Pallet Base", "Sticky Goo Pallet Base",
        "Nested Pallet Base", "Ceramic Marble Pallet Base",
    };
    const auto bulkPalletId = specIndex("bulk_pallet");
    const BodySpec& bulkPallet = specs_[bulkPalletId];
    const Real warehousePalletTop = bulkPallet.dimensions.y;
    for (int index = 0; index < kWarehousePalletBaseCount; ++index) {
        const Vec3 center = warehouseCenters[static_cast<std::size_t>(index)];
        spawn(bulkPalletId,
              {center.x, bulkPallet.dimensions.y * 0.5, center.z},
              std::string(baseLabels[static_cast<std::size_t>(index)]),
              index + 1, true, true, {},
              {0.48 + 0.025 * index, 0.30 + 0.012 * index, 0.13});
    }

    const auto spawnSphereStack = [&](std::uint16_t specId, const Vec3& center,
                                      int count, int across, int deep, Real gap,
                                      std::string_view label, int labelWidth,
                                      const auto& colors) {
        const BodySpec& spec = specs_[specId];
        const Real pitch = spec.diameter + gap;
        const int perLayer = across * deep;
        for (int index = 0; index < count; ++index) {
            const int layer = index / perLayer;
            const int slot = index % perLayer;
            const int row = slot / across;
            const int column = slot % across;
            spawn(specId,
                  {center.x + (column - (across - 1) * 0.5) * pitch,
                   warehousePalletTop + spec.radius() + layer * pitch,
                   center.z + (row - (deep - 1) * 0.5) * pitch},
                  numberedLabel(label, index + 1, count, labelWidth),
                  index + 1, true, true, {},
                  colors[static_cast<std::size_t>(index) % colors.size()]);
        }
    };

    constexpr std::array<Vec3, 6> bowlingColors{
        Vec3{0.10,0.24,0.72}, Vec3{0.50,0.10,0.68}, Vec3{0.08,0.62,0.58},
        Vec3{0.78,0.12,0.18}, Vec3{0.92,0.46,0.08}, Vec3{0.20,0.20,0.24},
    };
    constexpr std::array<Vec3, 6> dodgeColors{
        Vec3{0.95,0.18,0.16}, Vec3{0.16,0.45,0.96}, Vec3{0.98,0.78,0.12},
        Vec3{0.18,0.82,0.34}, Vec3{0.88,0.30,0.86}, Vec3{0.96,0.52,0.12},
    };
    constexpr std::array<Vec3, 4> bulkGooColors{
        Vec3{0.20,0.98,0.35}, Vec3{0.18,0.76,1.00},
        Vec3{0.92,0.28,0.96}, Vec3{0.98,0.72,0.18},
    };
    constexpr std::array<Vec3, 5> bulkMarbleColors{
        Vec3{0.96,0.95,0.89}, Vec3{0.54,0.78,0.98}, Vec3{0.98,0.70,0.35},
        Vec3{0.66,0.92,0.72}, Vec3{0.88,0.64,0.94},
    };
    spawnSphereStack(specIndex("bowling_ball"), warehouseCenters[0],
                     kPalletBowlingBallCount, 5, 3, 0.003,
                     "Bowling Ball", 3, bowlingColors);
    spawnSphereStack(specIndex("pallet_dodge"), warehouseCenters[1],
                     kPalletDodgeballCount, 5, 3, 0.003,
                     "Pallet Dodgeball", 3, dodgeColors);
    spawnSphereStack(specIndex("pallet_goo"), warehouseCenters[2],
                     kPalletGooCount, 10, 6, 0.003,
                     "Pallet Goo Glob", 3, bulkGooColors);
    spawnSphereStack(specIndex("pallet_marble"), warehouseCenters[4],
                     kPalletMarbleCount, 20, 15, 0.002,
                     "Pallet Marble", 4, bulkMarbleColors);

    for (int index = 0; index < kNestedPalletCount; ++index) {
        spawn(bulkPalletId,
              {warehouseCenters[3].x,
               warehousePalletTop + bulkPallet.dimensions.y * 0.5
                   + index * (bulkPallet.dimensions.y + 0.002),
               warehouseCenters[3].z},
              numberedLabel("Stacked EPAL Pallet", index + 1,
                            kNestedPalletCount, 2),
              kWarehousePalletBaseCount + index + 1, true, true, {},
              {0.50 + 0.012 * (index % 4), 0.31, 0.14});
    }

    const auto timberId = specIndex("timber_log");
    const BodySpec& timber = specs_[timberId];
    const Vec3 timberCenter{70.0, 0.0, 58.0};
    for (int index = 0; index < 3; ++index) {
        spawn(timberId,
              {timberCenter.x, timber.dimensions.y * 0.5,
               timberCenter.z + (index - 1) * 0.34},
              numberedLabel("Douglas-Fir Log", index + 1, kTimberLogCount),
              index + 1, true, true, {},
              {0.39 + 0.035 * index, 0.22 + 0.018 * index, 0.08});
    }
    for (int index = 0; index < 3; ++index) {
        spawn(timberId,
              {timberCenter.x + (index - 1) * 0.34,
               timber.dimensions.y * 1.5 + 0.005, timberCenter.z},
              numberedLabel("Douglas-Fir Log", index + 4, kTimberLogCount),
              index + 4, true, true, quarterYaw,
              {0.44 + 0.025 * index, 0.25 + 0.012 * index, 0.09});
    }

    std::vector<std::uint16_t> plushIds;
    for (std::size_t index = 0; index < specs_.size(); ++index) {
        if (specs_[index].category == "plush") {
            plushIds.push_back(static_cast<std::uint16_t>(index));
        }
    }
    std::shuffle(plushIds.begin(), plushIds.end(), rng_);
    for (int index = 0; index < kStuffedAnimalCount; ++index) {
        const auto plushId = plushIds[static_cast<std::size_t>(index)];
        spawn(plushId,
              {35.0 + index * 0.62, specs_[plushId].radius(), 16.0},
              numberedLabel(specs_[plushId].name, index + 1,
                            kStuffedAnimalCount),
              index + 1, true, true);
    }

    if (brickNumber != kClayBrickCount
        || bodies_.size() != static_cast<std::size_t>(kExpectedBodyCount)) {
        throw std::logic_error("scene population mismatch");
    }
    for (RigidBody& body : bodies_) {
        body.asleep = false;
        body.sleepTime = 0.0;
    }
    heldBody_ = -1;
    simulationTime_ = 0.0;
    lastBroadphaseCandidates_ = 0;
    lastActiveContacts_ = 0;
    lastSolverIterations_ = kSolverIterations;
    galileoState_ = GalileoExperimentState::Inactive;
    galileoBodyIds_ = {};
    galileoElapsed_ = 0.0;
    galileoSteelLandingTime_.reset();
    galileoConcreteLandingTime_.reset();
    galileoCurrentTrail_.clear();
    galileoPreviousTrail_.clear();
    gravityInterceptState_ = GravityInterceptState::Inactive;
    gravityInterceptBodyIds_ = {};
    gravityInterceptElapsed_ = 0.0;
    gravityInterceptHitTime_.reset();
    gravityInterceptHitPoint_.reset();
    gravityInterceptClosestGap_.reset();
    gravityInterceptCurrentTrail_.clear();
    gravityInterceptPreviousTrail_.clear();
    lastEchoPulseEvent_.reset();
    lastEchoPulseBodies_.clear();
    nextEchoPulseSerial_ = 1U;
    messages_.clear();
    addMessage("Full reset: Earth gravity, 0.65 friction, 1 N force, player start, and all 4,487 items awake.");
}

Real PhysicsWorld::dynamicMass(const RigidBody& body) const noexcept {
    return body.massCarriedByHost
        ? 0.0
        : specs_[body.spec].mass + specs_[body.spec].addedMass
            + body.attachedPayloadMass;
}

Real PhysicsWorld::inverseMass(const RigidBody& body) const noexcept {
    const Real mass = dynamicMass(body);
    return mass > 1.0e-12 ? 1.0 / mass : 0.0;
}

Real PhysicsWorld::supportExtent(const RigidBody& body,
                                 const Vec3& axis) const noexcept {
    const BodySpec& spec = specs_[body.spec];
    if (spec.shape == Shape::Sphere) return spec.radius();
    const Vec3 half = spec.halfExtents();
    const Vec3 local = body.orientation.conjugate().rotate(axis);
    return std::abs(local.x) * half.x + std::abs(local.y) * half.y
         + std::abs(local.z) * half.z;
}

Vec3 PhysicsWorld::supportPoint(const RigidBody& body,
                                const Vec3& direction) const noexcept {
    const BodySpec& spec = specs_[body.spec];
    if (spec.shape == Shape::Sphere) {
        return body.position
             + direction.normalized({1.0, 0.0, 0.0}) * spec.radius();
    }
    const Vec3 half = spec.halfExtents();
    const Vec3 localDirection = body.orientation.conjugate().rotate(direction);
    const auto coordinate = [](Real component, Real extent) {
        if (component > 1.0e-10) return extent;
        if (component < -1.0e-10) return -extent;
        return 0.0;
    };
    const Vec3 local{
        coordinate(localDirection.x, half.x),
        coordinate(localDirection.y, half.y),
        coordinate(localDirection.z, half.z),
    };
    return body.position + body.orientation.rotate(local);
}

void PhysicsWorld::wake(RigidBody& body, bool dirty) noexcept {
    body.asleep = false;
    body.sleepTime = 0.0;
    if (dirty) body.pristine = false;
}

void PhysicsWorld::wakeAll() {
    int activeBricks = 0;
    int activeBulkTotal = 0;
    std::array<int, kBulkCategories.size()> activeBulk{};
    for (const auto& body : bodies_) {
        if (specs_[body.spec].key == "clay_brick" && !body.asleep) {
            ++activeBricks;
        }
        const int family = bulkFamilyIndex(specs_[body.spec]);
        if (family >= 0 && !body.asleep) {
            ++activeBulkTotal;
            ++activeBulk[static_cast<std::size_t>(family)];
        }
    }
    int brickSlots = std::max(0, kMaximumActiveBricks - activeBricks);
    for (auto& body : bodies_) {
        const BodySpec& spec = specs_[body.spec];
        const std::string& key = spec.key;
        if (body.pristine && body.asleep
            && (key == "clay_brick" || key == "wood_pallet"
                || isBulkStock(spec))) {
            // A room-tuning change breaks the authored reset support. Wake a
            // bounded first cohort below; remaining stock is now dirty and
            // can join through normal staged contact propagation.
            body.pristine = false;
        }
        if (key == "clay_brick" && body.asleep) {
            if (brickSlots <= 0) continue;
            --brickSlots;
        }
        const int family = bulkFamilyIndex(spec);
        if (family >= 0 && body.asleep) {
            const std::size_t slot = static_cast<std::size_t>(family);
            if (activeBulkTotal >= kMaximumActiveBulkBodies
                || activeBulk[slot] >= kBulkFamilyCaps[slot]) {
                continue;
            }
            ++activeBulkTotal;
            ++activeBulk[slot];
        }
        wake(body);
    }
}

void PhysicsWorld::wakeNearby(const Vec3& position, Real radius, int limit) {
    std::vector<std::pair<Real, std::size_t>> nearby;
    const Real radius2 = radius * radius;
    for (std::size_t index = 0; index < bodies_.size(); ++index) {
        const Real distance2 = (bodies_[index].position - position).lengthSquared();
        if (distance2 <= radius2) nearby.emplace_back(distance2, index);
    }
    std::stable_sort(nearby.begin(), nearby.end());
    int activeBricks = 0;
    int activeBulkTotal = 0;
    std::array<int, kBulkCategories.size()> activeBulk{};
    for (const RigidBody& body : bodies_) {
        if (body.asleep) continue;
        if (specs_[body.spec].key == "clay_brick") ++activeBricks;
        const int family = bulkFamilyIndex(specs_[body.spec]);
        if (family >= 0) {
            ++activeBulkTotal;
            ++activeBulk[static_cast<std::size_t>(family)];
        }
    }
    const int wakeLimit = std::max(1, limit);
    int newlyWoken = 0;
    for (const auto& [distance2, bodyIndex] : nearby) {
        (void)distance2;
        if (newlyWoken >= wakeLimit) break;
        RigidBody& body = bodies_[bodyIndex];
        if (!body.asleep) continue;
        const BodySpec& spec = specs_[body.spec];
        if (spec.key == "clay_brick" && activeBricks >= kMaximumActiveBricks) {
            continue;
        }
        const int family = bulkFamilyIndex(spec);
        if (family >= 0) {
            const std::size_t slot = static_cast<std::size_t>(family);
            if (activeBulkTotal >= kMaximumActiveBulkBodies
                || activeBulk[slot] >= kBulkFamilyCaps[slot]) {
                continue;
            }
            ++activeBulkTotal;
            ++activeBulk[slot];
        }
        if (spec.key == "clay_brick") ++activeBricks;
        wake(body);
        ++newlyWoken;
    }
}

void PhysicsWorld::setMoveInput(Real forward, Real strafe) noexcept {
    if (playerPositionLocked_) {
        player_.moveForward = 0.0;
        player_.moveStrafe = 0.0;
        return;
    }
    player_.moveForward = clamp(forward, -1.0, 1.0);
    player_.moveStrafe = clamp(strafe, -1.0, 1.0);
}

void PhysicsWorld::togglePlayerPositionLock() {
    playerPositionLocked_ = !playerPositionLocked_;
    if (playerPositionLocked_) {
        lockedPlayerPosition_ = player_.position;
        player_.moveForward = 0.0;
        player_.moveStrafe = 0.0;
        player_.velocity = {};
        player_.landingSpeed = 0.0;
        player_.previousPosition = lockedPlayerPosition_;
        std::ostringstream stream;
        stream << "Player position locked at (" << std::fixed << std::setprecision(2)
               << lockedPlayerPosition_.x << ", " << lockedPlayerPosition_.y
               << ", " << lockedPlayerPosition_.z
               << "). Camera, pickup, and throwing remain active.";
        addMessage(stream.str());
    } else {
        player_.position = lockedPlayerPosition_;
        player_.previousPosition = lockedPlayerPosition_;
        player_.velocity = {};
        player_.landingSpeed = 0.0;
        player_.grounded = player_.position.y <= 1.0e-9;
        addMessage("Player position unlocked. Full physics movement restored.");
    }
}

void PhysicsWorld::releaseHeldBodyForExperiment() noexcept {
    if (heldBody_ < 0 || heldBody_ >= static_cast<int>(bodies_.size())) return;
    RigidBody& held = bodies_[static_cast<std::size_t>(heldBody_)];
    held.held = false;
    wake(held);
    heldBody_ = -1;
}

void PhysicsWorld::detachPayloadsFromExperimentBodies(
    std::span<const int> ids) {
    for (RigidBody& body : bodies_) {
        if (std::find(ids.begin(), ids.end(), body.stuckTo) == ids.end()) {
            continue;
        }
        body.stuckTo = -1;
        body.massCarriedByHost = false;
        wake(body);
    }
}

void PhysicsWorld::prepareExperimentBody(int id, const Vec3& position,
                                         const Vec3& velocity) {
    if (id < 0 || id >= static_cast<int>(bodies_.size())) {
        throw std::out_of_range("experiment body index outside world");
    }
    RigidBody& body = bodies_[static_cast<std::size_t>(id)];
    if (body.stuckTo >= 0
        && body.stuckTo < static_cast<int>(bodies_.size())) {
        RigidBody& previousHost =
            bodies_[static_cast<std::size_t>(body.stuckTo)];
        const BodySpec& bodySpec = specs_[body.spec];
        previousHost.attachedPayloadMass = std::max(
            0.0, previousHost.attachedPayloadMass
                - (bodySpec.mass + bodySpec.addedMass
                   + body.attachedPayloadMass));
    }
    body.position = position;
    body.previousPosition = position;
    body.velocity = velocity;
    body.orientation = {};
    body.previousOrientation = {};
    body.angularVelocity = {};
    body.force = {};
    body.torque = {};
    body.stuckNormal = {};
    body.stuckLocalPosition = {};
    body.sleepTime = 0.0;
    body.impactCooldown = 0.0;
    body.lastImpulse = 0.0;
    body.attachedPayloadMass = 0.0;
    body.stuckTo = -1;
    body.asleep = false;
    body.grounded = false;
    body.held = false;
    body.stuckSurface = false;
    body.pristine = false;
    body.massCarriedByHost = false;
}

void PhysicsWorld::startGalileoExperiment() {
    const auto findBody = [&](std::string_view key) {
        for (std::size_t index = 0; index < bodies_.size(); ++index) {
            if (specs_[bodies_[index].spec].key == key) {
                return static_cast<int>(index);
            }
        }
        throw std::logic_error("Galileo experiment body missing: "
                               + std::string(key));
    };

    const GalileoBodyIds nextIds{findBody("steel"), findBody("concrete")};
    if (nextIds.steel == nextIds.concrete) {
        throw std::logic_error("Galileo experiment requires two distinct bodies");
    }

    if (!galileoCurrentTrail_.empty()) {
        galileoPreviousTrail_ = galileoCurrentTrail_;
    }
    galileoCurrentTrail_.clear();
    galileoBodyIds_ = nextIds;
    galileoElapsed_ = 0.0;
    galileoSteelLandingTime_.reset();
    galileoConcreteLandingTime_.reset();

    // If adhesive cargo was added during free play, release it before reusing
    // either calibration ball so the two dynamic masses again match their
    // catalog values. This does not reposition or otherwise reset the cargo.
    const std::array<int, 2> experimentIds{{nextIds.steel, nextIds.concrete}};
    detachPayloadsFromExperimentBodies(experimentIds);
    // F6 stages the observer at a dedicated viewing position.  Release any
    // held prop first so its spring constraint cannot chase that teleport and
    // inject an unrelated impulse into the experiment.
    releaseHeldBodyForExperiment();

    prepareExperimentBody(nextIds.steel, kGalileoSteelDropPosition);
    prepareExperimentBody(nextIds.concrete, kGalileoConcreteDropPosition);

    // F6 may stage the player immediately after this call.  Clearing the lock
    // prevents the end-of-step invariant from snapping that camera back to an
    // earlier anchor; ordinary movement remains enabled throughout the run.
    playerPositionLocked_ = false;
    lockedPlayerPosition_ = player_.position;
    galileoState_ = GalileoExperimentState::Running;
    galileoCurrentTrail_.push_back({0.0, kGalileoSteelDropPosition,
                                    kGalileoConcreteDropPosition});
    addMessage("Galileo vacuum drop started: steel and concrete released together at 7.00 m.");
}

void PhysicsWorld::startGravityInterceptExperiment() {
    const auto findProjectile = [&] {
        for (std::size_t index = 0; index < bodies_.size(); ++index) {
            if (specs_[bodies_[index].spec].key == "ceramic_marble") {
                return static_cast<int>(index);
            }
        }
        throw std::logic_error(
            "gravity intercept projectile missing: ceramic_marble");
    };
    const auto findTarget = [&] {
        for (std::size_t index = 0; index < bodies_.size(); ++index) {
            if (specs_[bodies_[index].spec].category == "plush") {
                return static_cast<int>(index);
            }
        }
        throw std::logic_error("gravity intercept target missing: plush");
    };

    const GravityInterceptBodyIds nextIds{findProjectile(), findTarget()};
    if (nextIds.projectile == nextIds.target) {
        throw std::logic_error(
            "gravity intercept requires two distinct bodies");
    }

    if (!gravityInterceptCurrentTrail_.empty()) {
        gravityInterceptPreviousTrail_ = gravityInterceptCurrentTrail_;
    }
    gravityInterceptCurrentTrail_.clear();
    gravityInterceptBodyIds_ = nextIds;
    gravityInterceptElapsed_ = 0.0;
    gravityInterceptHitTime_.reset();
    gravityInterceptHitPoint_.reset();
    gravityInterceptClosestGap_.reset();

    const std::array<int, 2> experimentIds{{
        nextIds.projectile, nextIds.target,
    }};
    detachPayloadsFromExperimentBodies(experimentIds);
    releaseHeldBodyForExperiment();

    const Vec3 launchVelocity =
        (kGravityInterceptTargetPosition
         - kGravityInterceptProjectilePosition)
            .normalized({1.0, 0.0, 0.0})
        * kGravityInterceptLaunchSpeed;
    prepareExperimentBody(nextIds.projectile,
                          kGravityInterceptProjectilePosition,
                          launchVelocity);
    prepareExperimentBody(nextIds.target, kGravityInterceptTargetPosition);

    const RigidBody& projectile =
        bodies_[static_cast<std::size_t>(nextIds.projectile)];
    const RigidBody& target =
        bodies_[static_cast<std::size_t>(nextIds.target)];
    const Real radii = specs_[projectile.spec].radius()
                     + specs_[target.spec].radius();
    gravityInterceptClosestGap_ = std::max(
        0.0, (target.position - projectile.position).length() - radii);
    gravityInterceptState_ = GravityInterceptState::Running;
    gravityInterceptCurrentTrail_.push_back({
        0.0, kGravityInterceptProjectilePosition,
        kGravityInterceptTargetPosition,
    });

    // F8 stages a broadside observer. As with F6, clear a previous player lock
    // so the next end-of-step invariant cannot snap that staged camera away.
    playerPositionLocked_ = false;
    lockedPlayerPosition_ = player_.position;
    const BodySpec& targetSpec = specs_[target.spec];
    const std::string targetName = target.instanceLabel.empty()
        ? targetSpec.name : target.instanceLabel;
    addMessage("Gravity-proof trick shot started: ceramic marble aimed at "
               + targetName + " in a local vacuum.");
}

std::string_view PhysicsWorld::galileoStatus() const noexcept {
    switch (galileoState_) {
    case GalileoExperimentState::Inactive:
        return "Galileo experiment ready";
    case GalileoExperimentState::Running:
        return "Galileo vacuum drop running";
    case GalileoExperimentState::Complete:
        return "Galileo vacuum drop complete";
    }
    return "Galileo experiment ready";
}

bool PhysicsWorld::isGalileoBody(std::size_t index) const noexcept {
    return (galileoBodyIds_.steel >= 0
            && index == static_cast<std::size_t>(galileoBodyIds_.steel))
        || (galileoBodyIds_.concrete >= 0
            && index == static_cast<std::size_t>(galileoBodyIds_.concrete));
}

std::string_view PhysicsWorld::gravityInterceptStatus() const noexcept {
    switch (gravityInterceptState_) {
    case GravityInterceptState::Inactive:
        return "Gravity intercept ready";
    case GravityInterceptState::Running:
        return "Gravity-proof trick shot running";
    case GravityInterceptState::Hit:
        return "Gravity-proof trick shot hit";
    case GravityInterceptState::Missed:
        return "Gravity-proof trick shot missed";
    }
    return "Gravity intercept ready";
}

bool PhysicsWorld::isGravityInterceptBody(std::size_t index) const noexcept {
    return (gravityInterceptBodyIds_.projectile >= 0
            && index == static_cast<std::size_t>(
                gravityInterceptBodyIds_.projectile))
        || (gravityInterceptBodyIds_.target >= 0
            && index == static_cast<std::size_t>(
                gravityInterceptBodyIds_.target));
}

bool PhysicsWorld::isIdealVacuumBody(std::size_t index) const noexcept {
    return (galileoState_ == GalileoExperimentState::Running
            && isGalileoBody(index))
        || (gravityInterceptState_ == GravityInterceptState::Running
            && isGravityInterceptBody(index));
}

void PhysicsWorld::updateGalileoExperiment(Real dt) {
    if (galileoState_ != GalileoExperimentState::Running) return;
    if (galileoBodyIds_.steel < 0 || galileoBodyIds_.concrete < 0
        || static_cast<std::size_t>(galileoBodyIds_.steel) >= bodies_.size()
        || static_cast<std::size_t>(galileoBodyIds_.concrete) >= bodies_.size()) {
        throw std::logic_error("Galileo experiment body index became invalid");
    }

    galileoElapsed_ += dt;
    const RigidBody& steel = bodies_[static_cast<std::size_t>(galileoBodyIds_.steel)];
    const RigidBody& concrete =
        bodies_[static_cast<std::size_t>(galileoBodyIds_.concrete)];
    if (galileoCurrentTrail_.size() < kGalileoMaxTrailSamples) {
        galileoCurrentTrail_.push_back({galileoElapsed_, steel.position,
                                       concrete.position});
    }

    // advanceBodySwept marks the first floor hit even when restitution carries
    // the body above the surface during the remainder of that same tick.
    // Position-only tests would therefore report the later return from the
    // bounce and make material restitution look like unequal fall time.
    const auto landed = [](const RigidBody& body) { return body.grounded; };
    if (!galileoSteelLandingTime_ && landed(steel)) {
        galileoSteelLandingTime_ = galileoElapsed_;
    }
    if (!galileoConcreteLandingTime_ && landed(concrete)) {
        galileoConcreteLandingTime_ = galileoElapsed_;
    }
    if (!galileoSteelLandingTime_ || !galileoConcreteLandingTime_) return;

    galileoState_ = GalileoExperimentState::Complete;
    std::ostringstream stream;
    stream << "Galileo vacuum drop complete: steel " << std::fixed
           << std::setprecision(3) << *galileoSteelLandingTime_
           << " s, concrete " << *galileoConcreteLandingTime_ << " s.";
    addMessage(stream.str());
}

void PhysicsWorld::updateGravityInterceptExperiment(Real dt) {
    if (gravityInterceptState_ != GravityInterceptState::Running) return;
    const int projectileId = gravityInterceptBodyIds_.projectile;
    const int targetId = gravityInterceptBodyIds_.target;
    if (projectileId < 0 || targetId < 0 || projectileId == targetId
        || projectileId >= static_cast<int>(bodies_.size())
        || targetId >= static_cast<int>(bodies_.size())) {
        throw std::logic_error(
            "gravity intercept body index became invalid");
    }

    const double stepStart = gravityInterceptElapsed_;
    gravityInterceptElapsed_ += dt;
    const RigidBody& projectile =
        bodies_[static_cast<std::size_t>(projectileId)];
    const RigidBody& target = bodies_[static_cast<std::size_t>(targetId)];
    if (gravityInterceptCurrentTrail_.size()
        < kGravityInterceptMaxTrailSamples) {
        gravityInterceptCurrentTrail_.push_back({
            gravityInterceptElapsed_, projectile.position, target.position,
        });
    }

    const Real radii = specs_[projectile.spec].radius()
                     + specs_[target.spec].radius();
    const Real gap = std::max(
        0.0, (target.position - projectile.position).length() - radii);
    gravityInterceptClosestGap_ = gravityInterceptClosestGap_
        ? std::min(*gravityInterceptClosestGap_, gap) : gap;

    const std::uint16_t projectileIndex =
        static_cast<std::uint16_t>(projectileId);
    const std::uint16_t targetIndex = static_cast<std::uint16_t>(targetId);
    const auto matchingContact = std::find_if(
        contacts_.begin(), contacts_.end(), [&](const Contact& contact) {
            return (contact.first == projectileIndex
                    && contact.second == targetIndex)
                || (contact.first == targetIndex
                    && contact.second == projectileIndex);
        });
    if (matchingContact != contacts_.end()) {
        gravityInterceptState_ = GravityInterceptState::Hit;
        gravityInterceptHitTime_ = stepStart
            + dt * clamp(matchingContact->toi, 0.0, 1.0);
        gravityInterceptHitPoint_ = matchingContact->point;
        gravityInterceptClosestGap_ = 0.0;
        std::ostringstream stream;
        stream << "Gravity-proof trick shot hit at " << std::fixed
               << std::setprecision(4) << *gravityInterceptHitTime_
               << " s; the shared gravity preserved the firing line.";
        addMessage(stream.str());
        return;
    }

    const bool obstructed = std::any_of(
        contacts_.begin(), contacts_.end(), [&](const Contact& contact) {
            return contact.first == projectileIndex
                || contact.second == projectileIndex
                || contact.first == targetIndex
                || contact.second == targetIndex;
        });
    if (!obstructed && !projectile.grounded && !target.grounded
        && gravityInterceptElapsed_ < kGravityInterceptTimeout) {
        return;
    }

    gravityInterceptState_ = GravityInterceptState::Missed;
    std::ostringstream stream;
    stream << "Gravity-proof trick shot missed; closest surface gap "
           << std::fixed << std::setprecision(4)
           << gravityInterceptClosestGap_.value_or(0.0) << " m.";
    addMessage(stream.str());
}

void PhysicsWorld::adjustGravity(int direction) {
    const Real old = gravity_;
    gravity_ = clamp(roundedTo(gravity_ + direction * kGravityStep, 100'000.0),
                     kGravityMin, kGravityMax);
    if (std::abs(old - gravity_) > 1.0e-12) {
        wakeAll();
        std::ostringstream stream;
        stream << "Gravity set to " << std::setprecision(6) << gravity_ << " m/s^2.";
        addMessage(stream.str());
    }
}

void PhysicsWorld::adjustFriction(int direction) {
    const Real old = roomFriction_;
    roomFriction_ = clamp(roundedTo(roomFriction_ + direction * kFrictionStep, 100.0),
                          kFrictionMin, kFrictionMax);
    if (std::abs(old - roomFriction_) > 1.0e-12) {
        wakeAll();
        std::ostringstream stream;
        stream << "Room friction coefficient set to " << std::fixed
               << std::setprecision(2) << roomFriction_ << '.';
        addMessage(stream.str());
    }
}

void PhysicsWorld::adjustThrowForce(int direction) {
    static constexpr std::array<Real, 19> forces{
        1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
        1'000.0, 2'000.0, 5'000.0, 10'000.0, 20'000.0, 50'000.0,
        100'000.0, 200'000.0, 500'000.0, 1'000'000.0,
    };
    auto nearest = std::min_element(forces.begin(), forces.end(), [&](Real a, Real b) {
        return std::abs(a - throwForce_) < std::abs(b - throwForce_);
    });
    const auto current = static_cast<int>(std::distance(forces.begin(), nearest));
    const int selected = clamp(current + direction, 0,
                               static_cast<int>(forces.size()) - 1);
    throwForce_ = forces[static_cast<std::size_t>(selected)];
    std::ostringstream stream;
    stream << "Throw force set to " << std::fixed << std::setprecision(0)
           << throwForce_ << " N.";
    addMessage(stream.str());
}

bool PhysicsWorld::jump() {
    if (playerPositionLocked_) {
        addMessage("Position lock is active; press B to restore physics movement.");
        return false;
    }
    if (!player_.grounded) return false;
    player_.velocity.y = std::sqrt(2.0 * gravity_ * kJumpHeight);
    player_.grounded = false;
    addMessage("Jump impulse set for a 1.0 m apex at current gravity.");
    return true;
}

int PhysicsWorld::raycastBody(const Vec3& origin, const Vec3& rawDirection,
                              Real reach) const {
    const Vec3 direction = rawDirection.normalized({0.0, 0.0, 1.0});
    int best = -1;
    Real bestDistance = reach + 1.0;
    for (std::size_t index = 0; index < bodies_.size(); ++index) {
        const RigidBody& body = bodies_[index];
        if (body.held) continue;
        const Real radius = body.cachedBoundingRadius
            * (specs_[body.spec].shape == Shape::Sphere ? 1.0 : 1.15);
        const Vec3 offset = origin - body.position;
        const Real b = offset.dot(direction);
        const Real c = offset.lengthSquared() - radius * radius;
        const Real discriminant = b * b - c;
        if (discriminant < 0.0) continue;
        const Real root = std::sqrt(discriminant);
        Real distance = -b - root;
        if (distance < 0.0) distance = -b + root;
        if (distance >= 0.0 && distance <= reach && distance < bestDistance) {
            bestDistance = distance;
            best = static_cast<int>(index);
        }
    }
    return best;
}

double PhysicsWorld::echoPulseCooldownRemaining() const noexcept {
    if (!lastEchoPulseEvent_) return 0.0;
    return std::max(0.0, lastEchoPulseEvent_->simulationTime
                          + kEchoPulseCooldown - simulationTime_);
}

std::string_view PhysicsWorld::echoPulseStatus() const noexcept {
    return echoPulseReady() ? std::string_view{"Echo pulse ready"}
                            : std::string_view{"Echo pulse recharging"};
}

bool PhysicsWorld::emitEchoPulse() {
    const double cooldown = echoPulseCooldownRemaining();
    if (cooldown > 0.0) {
        std::ostringstream stream;
        stream << "Echo pulse recharging: " << std::fixed
               << std::setprecision(2) << cooldown << " s remaining.";
        addMessage(stream.str());
        return false;
    }

    const Vec3 source = player_.eye();
    const Vec3 direction = player_.forward().normalized({0.0, 0.0, 1.0});
    const int target = raycastBody(source, direction, kEchoPulseReach);
    const Vec3 origin = target >= 0
        ? bodies_[static_cast<std::size_t>(target)].position
        : source + direction * kEchoPulseFallbackDistance;

    struct Candidate {
        std::uint16_t body{};
        Real surfaceDistance{};
    };
    std::array<Candidate, kEchoPulseMaximumAffectedBodies> candidates{};
    std::size_t candidateCount = 0U;
    const auto candidateLess = [](const Candidate& first,
                                  const Candidate& second) noexcept {
        return first.surfaceDistance < second.surfaceDistance
            || (first.surfaceDistance == second.surfaceDistance
                && first.body < second.body);
    };

    // Retain only the nearest fixed-size cohort. This turns a pulse aimed into
    // a 3,000-marble pallet into O(body count) selection with no allocation and
    // a hard upper bound on newly active solver work.
    for (std::size_t index = 0; index < bodies_.size(); ++index) {
        const RigidBody& body = bodies_[index];
        if (body.held || static_cast<int>(index) == heldBody_) continue;
        const Real centerDistance = (body.position - origin).length();
        const Real surfaceDistance = std::max(
            0.0, centerDistance - body.cachedBoundingRadius);
        if (surfaceDistance > kEchoPulseRadius) continue;
        const Candidate candidate{static_cast<std::uint16_t>(index),
                                  surfaceDistance};
        if (candidateCount < candidates.size()) {
            candidates[candidateCount++] = candidate;
            continue;
        }
        const auto worst = std::max_element(candidates.begin(), candidates.end(),
                                            candidateLess);
        if (candidateLess(candidate, *worst)) *worst = candidate;
    }
    if (target >= 0) {
        const auto targetBody = static_cast<std::uint16_t>(target);
        const bool retained = std::any_of(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(candidateCount),
            [targetBody](const Candidate& candidate) {
                return candidate.body == targetBody;
            });
        if (!retained) {
            const Candidate targetCandidate{targetBody, 0.0};
            if (candidateCount < candidates.size()) {
                candidates[candidateCount++] = targetCandidate;
            } else {
                const auto worst = std::max_element(
                    candidates.begin(), candidates.end(), candidateLess);
                *worst = targetCandidate;
            }
        }
    }
    std::sort(candidates.begin(), candidates.begin()
                                  + static_cast<std::ptrdiff_t>(candidateCount),
              candidateLess);

    // A pulse can pull adhesive payloads free, but attachment bookkeeping is
    // completed before any mass-aware impulse is calculated. Thus a host's
    // impulse never depends on whether its child happens to sort before it.
    for (std::size_t slot = 0; slot < candidateCount; ++slot) {
        RigidBody& body = bodies_[candidates[slot].body];
        if (body.stuckTo >= 0
            && body.stuckTo < static_cast<int>(bodies_.size())) {
            RigidBody& host = bodies_[static_cast<std::size_t>(body.stuckTo)];
            const BodySpec& spec = specs_[body.spec];
            const Real transferred = spec.mass + spec.addedMass
                                   + body.attachedPayloadMass;
            host.attachedPayloadMass = std::max(
                0.0, host.attachedPayloadMass - transferred);
            body.stuckTo = -1;
            body.massCarriedByHost = false;
        }
        body.stuckSurface = false;
    }

    lastEchoPulseBodies_.clear();
    Real totalDeliveredImpulse = 0.0;
    for (std::size_t slot = 0; slot < candidateCount; ++slot) {
        const Candidate candidate = candidates[slot];
        RigidBody& body = bodies_[candidate.body];
        const Real mass = dynamicMass(body);
        if (mass <= 1.0e-12) continue;
        const Real normalizedDistance = clamp(
            candidate.surfaceDistance / kEchoPulseRadius, 0.0, 1.0);
        const Real proximity = 1.0 - normalizedDistance;
        const Real smoothFalloff = proximity * proximity
                                 * (3.0 - 2.0 * proximity);
        const Real desiredDeltaSpeed = clamp(
            4.8 / std::sqrt(std::max(mass, 0.25)), 0.55, 8.0);
        const Real impulseMagnitude = std::min(
            kEchoPulseMaximumImpulse, mass * desiredDeltaSpeed) * smoothFalloff;
        if (impulseMagnitude <= 1.0e-9) continue;
        const Vec3 radial = (body.position - origin).normalized(direction);
        const Vec3 impulse = radial * impulseMagnitude;
        const Vec3 eventPosition = body.position;
        applyImpulseRaw(specs_[body.spec], body, inverseMass(body), impulse,
                        body.position);
        wake(body);
        body.lastImpulse = std::max(body.lastImpulse, impulseMagnitude);
        lastEchoPulseBodies_.push_back(
            {candidate.body, eventPosition, impulse,
             candidate.surfaceDistance});
        totalDeliveredImpulse += impulseMagnitude;
    }

    if (!playerPositionLocked_) {
        const Real recoilSpeed = 0.18
            + std::min(0.52, totalDeliveredImpulse * 0.003);
        player_.velocity -= direction * recoilSpeed;
    }

    EchoPulseEvent event;
    event.serial = nextEchoPulseSerial_++;
    if (nextEchoPulseSerial_ == 0U) nextEchoPulseSerial_ = 1U;
    event.source = source;
    event.origin = origin;
    event.direction = direction;
    event.simulationTime = simulationTime_;
    event.radius = kEchoPulseRadius;
    event.targetBody = target;
    event.affectedBodyCount = static_cast<std::uint32_t>(
        lastEchoPulseBodies_.size());
    event.totalDeliveredImpulse = totalDeliveredImpulse;
    lastEchoPulseEvent_ = event;

    std::ostringstream stream;
    stream << "Echo pulse #" << event.serial << (target >= 0 ? " locked: " : ": ")
           << event.affectedBodyCount << " bodies, " << std::fixed
           << std::setprecision(2) << totalDeliveredImpulse
           << " N*s delivered.";
    addMessage(stream.str());
    return true;
}

bool PhysicsWorld::pickupOrDrop() {
    if (heldBody_ >= 0) {
        RigidBody& body = bodies_[static_cast<std::size_t>(heldBody_)];
        body.held = false;
        wake(body);
        addMessage("Released " + (body.instanceLabel.empty()
            ? specs_[body.spec].name : body.instanceLabel) + " without a throw.");
        heldBody_ = -1;
        return true;
    }
    const int selected = raycastBody(player_.eye(), player_.forward());
    if (selected < 0) {
        addMessage("No item is within the crosshair's 6 m pickup reach.");
        return false;
    }
    RigidBody& body = bodies_[static_cast<std::size_t>(selected)];
    body.held = true;
    body.stuckSurface = false;
    if (body.stuckTo >= 0) {
        RigidBody& host = bodies_[static_cast<std::size_t>(body.stuckTo)];
        host.attachedPayloadMass = std::max(
            0.0, host.attachedPayloadMass
                - (specs_[body.spec].mass + specs_[body.spec].addedMass
                   + body.attachedPayloadMass));
        body.stuckTo = -1;
        body.massCarriedByHost = false;
    }
    wake(body);
    const BodySpec& selectedSpec = specs_[body.spec];
    if (selectedSpec.key == "clay_brick" || selectedSpec.key == "wood_pallet") {
        wakeNearby(body.position, 0.16, kMaximumActiveBricks);
    } else if (isBulkStock(selectedSpec)) {
        wakeNearby(body.position,
                   std::max(0.18, body.cachedBoundingRadius * 2.4),
                   16);
    }
    heldBody_ = selected;
    addMessage("Holding " + (body.instanceLabel.empty()
        ? specs_[body.spec].name : body.instanceLabel) + '.');
    return true;
}

bool PhysicsWorld::throwHeld() {
    if (heldBody_ < 0) {
        addMessage("Pick up an item with right mouse before throwing.");
        return false;
    }
    RigidBody& body = bodies_[static_cast<std::size_t>(heldBody_)];
    const Real bodyMass = dynamicMass(body);
    if (bodyMass <= 1.0e-12) return false;
    const Vec3 direction = player_.forward().normalized({0.0, 0.0, 1.0});
    const Real reducedMass = bodyMass * kPlayerMass / (bodyMass + kPlayerMass);
    const Real impulseMagnitude = std::sqrt(
        2.0 * throwForce_ * kThrowStroke * reducedMass);
    const Vec3 impulse = direction * impulseMagnitude;
    body.held = false;
    body.velocity = player_.velocity;
    applyImpulseRaw(specs_[body.spec], body, inverseMass(body), impulse, body.position);
    wake(body);
    if (!playerPositionLocked_) {
        player_.velocity -= impulse / kPlayerMass;
    }
    const std::string name = body.instanceLabel.empty()
        ? specs_[body.spec].name : body.instanceLabel;
    heldBody_ = -1;
    std::ostringstream stream;
    stream << "Threw " << name << ": " << std::fixed << std::setprecision(0)
           << throwForce_ << " N across " << std::setprecision(2)
           << kThrowStroke << " m, impulse " << impulseMagnitude << " N*s.";
    addMessage(stream.str());
    return true;
}

std::string_view PhysicsWorld::lastMessage() const noexcept {
    return messages_.empty() ? std::string_view{} : std::string_view(messages_.back());
}

void PhysicsWorld::addMessage(std::string message) {
    if (messages_.size() >= 256) messages_.pop_front();
    messages_.push_back(std::move(message));
}

Real PhysicsWorld::effectiveRoomFriction(const RigidBody& body) const noexcept {
    return kDefaultRoomFriction > 0.0
        ? specs_[body.spec].friction * roomFriction_ / kDefaultRoomFriction
        : specs_[body.spec].friction;
}

Vec3 PhysicsWorld::bodyDrag(const RigidBody& body) const noexcept {
    const Real speed = body.velocity.length();
    if (speed < 1.0e-5) return {};
    const BodySpec& spec = specs_[body.spec];
    const Real magnitude = 0.5 * kAirDensity * spec.dragCoefficient
                         * spec.frontalArea() * speed * speed;
    return body.velocity * (-magnitude / speed);
}

void PhysicsWorld::enforcePlayerPositionLock() noexcept {
    if (!playerPositionLocked_) return;
    player_.position = lockedPlayerPosition_;
    player_.previousPosition = lockedPlayerPosition_;
    player_.velocity = {};
    player_.moveForward = 0.0;
    player_.moveStrafe = 0.0;
    player_.landingSpeed = 0.0;
}

void PhysicsWorld::integratePlayer(Real dt) {
    if (playerPositionLocked_) {
        enforcePlayerPositionLock();
        return;
    }
    player_.previousPosition = player_.position;
    const bool wasGrounded = player_.grounded;
    Vec3 desired = player_.forward(false) * player_.moveForward
                 + player_.right() * player_.moveStrafe;
    if (desired.lengthSquared() > 1.0) desired = desired.normalized();
    desired *= kPlayerWalkSpeed;
    const Real acceleration = player_.grounded
        ? kPlayerGroundAcceleration : kPlayerAirAcceleration;
    if (desired.lengthSquared() > 1.0e-8) {
        player_.velocity.x = approach(player_.velocity.x, desired.x,
                                      acceleration * dt);
        player_.velocity.z = approach(player_.velocity.z, desired.z,
                                      acceleration * dt);
    } else if (player_.grounded) {
        const Real speed = player_.velocity.horizontal().length();
        if (speed > 1.0e-8) {
            const Real next = std::max(0.0, speed - roomFriction_ * gravity_ * dt);
            const Real scale = next / speed;
            player_.velocity.x *= scale;
            player_.velocity.z *= scale;
        }
    }

    const Real verticalBefore = player_.velocity.y;
    player_.position.x += player_.velocity.x * dt;
    player_.position.z += player_.velocity.z * dt;
    player_.position.y += verticalBefore * dt - 0.5 * gravity_ * dt * dt;
    player_.velocity.y = verticalBefore - gravity_ * dt;

    if (player_.position.x < kPlayerRadius) {
        player_.position.x = kPlayerRadius;
        if (player_.velocity.x < 0.0) player_.velocity.x = 0.0;
    } else if (player_.position.x > kRoomWidth - kPlayerRadius) {
        player_.position.x = kRoomWidth - kPlayerRadius;
        if (player_.velocity.x > 0.0) player_.velocity.x = 0.0;
    }
    if (player_.position.z < kPlayerRadius) {
        player_.position.z = kPlayerRadius;
        if (player_.velocity.z < 0.0) player_.velocity.z = 0.0;
    } else if (player_.position.z > kRoomLength - kPlayerRadius) {
        player_.position.z = kRoomLength - kPlayerRadius;
        if (player_.velocity.z > 0.0) player_.velocity.z = 0.0;
    }
    const Real maximumFeet = kRoomHeight - kPlayerHeight;
    if (player_.position.y <= 0.0) {
        if (!wasGrounded && player_.velocity.y < 0.0) {
            player_.landingSpeed = -player_.velocity.y;
        }
        player_.position.y = 0.0;
        player_.velocity.y = 0.0;
        player_.grounded = true;
    } else {
        player_.grounded = false;
    }
    if (player_.position.y > maximumFeet) {
        player_.position.y = maximumFeet;
        if (player_.velocity.y > 0.0) player_.velocity.y *= -0.05;
    }
}

void PhysicsWorld::holdConstraint(RigidBody& body) {
    const BodySpec& spec = specs_[body.spec];
    const Real mass = dynamicMass(body);
    if (spec.key == "wheelbarrow") {
        const Vec3 forward = player_.forward(false).normalized({0.0, 0.0, 1.0});
        Vec3 target = player_.position + forward * 1.38;
        target.y = spec.dimensions.y * 0.5;
        const Vec3 error = target - body.position;
        const Vec3 relative = body.velocity - player_.velocity;
        Vec3 desiredAcceleration = error * 25.0 - relative * 10.0;
        desiredAcceleration.y = 0.0;
        Vec3 grip = desiredAcceleration * mass;
        const Real magnitude = grip.length();
        if (magnitude > 600.0) grip *= 600.0 / magnitude;
        body.force += grip;
        const Vec3 currentForward = body.orientation.rotate({0.0, 0.0, 1.0})
            .horizontal().normalized({0.0, 0.0, 1.0});
        const Real yawError = currentForward.cross(forward).y;
        body.torque.y += yawError * 420.0 - body.angularVelocity.y * 58.0;
        if (!playerPositionLocked_) {
            player_.velocity -= grip * (kFixedDt / kPlayerMass);
        }
        return;
    }
    Vec3 target = player_.eye() + player_.forward() * kHoldDistance;
    const Real radius = body.cachedBoundingRadius;
    target.x = clamp(target.x, radius + 0.02, kRoomWidth - radius - 0.02);
    target.y = clamp(target.y, radius + 0.02, kRoomHeight - radius - 0.02);
    target.z = clamp(target.z, radius + 0.02, kRoomLength - radius - 0.02);
    const Vec3 error = target - body.position;
    const Vec3 relative = body.velocity - player_.velocity;
    constexpr Real omega = 18.0;
    const Vec3 desiredAcceleration = error * (omega * omega)
                                   - relative * (2.0 * omega);
    Vec3 grip = desiredAcceleration * mass;
    const Real maximumGrip = std::max(2'500.0, mass * 250.0);
    const Real magnitude = grip.length();
    if (magnitude > maximumGrip) grip *= maximumGrip / magnitude;
    body.force += grip;
    if (!playerPositionLocked_) {
        player_.velocity -= grip * (kFixedDt / kPlayerMass);
    }
}

void PhysicsWorld::integrateBody(std::size_t index, Real dt) {
    RigidBody& body = bodies_[index];
    const BodySpec& spec = specs_[body.spec];
    if (body.stuckSurface) {
        const Real separating = std::max(0.0, body.force.dot(body.stuckNormal));
        if (separating <= spec.adhesionStrength) {
            body.force = {};
            body.torque = {};
            return;
        }
        body.stuckSurface = false;
        body.asleep = false;
    }
    // Body-adhered goo is a mass-carrying child. Pickup detaches it before an
    // external hand force can be applied, so normal integration only follows
    // the host transform.
    if (body.stuckTo >= 0) {
        body.force = {};
        body.torque = {};
        return;
    }
    if (body.asleep && !body.held) {
        body.force = {};
        body.torque = {};
        return;
    }
    // Reset-authored pallet loads begin in supported, strapped equilibrium.
    // They are still awake and participate in contacts/raycasting, but do not
    // manufacture a gravity-driven collapse from tiny visual packing gaps.
    // Pickup, throws, tuning changes, or a dirty impact clear `pristine` and
    // release the ordinary unconstrained rigid-body dynamics immediately.
    if (body.pristine && body.grounded && !body.held
        && body.velocity.lengthSquared() <= 1.0e-12
        && body.angularVelocity.lengthSquared() <= 1.0e-12
        && body.force.lengthSquared() <= 1.0e-12
        && body.torque.lengthSquared() <= 1.0e-12) {
        body.previousPosition = body.position;
        body.previousOrientation = body.orientation;
        return;
    }
    body.previousPosition = body.position;
    body.previousOrientation = body.orientation;
    const bool idealVacuum = isIdealVacuumBody(index);
    const Vec3 gravityForce{0.0, -spec.mass * gravity_, 0.0};
    const Vec3 buoyancy = idealVacuum
        ? Vec3{}
        : Vec3{0.0, kAirDensity * spec.buoyancyVolume * gravity_, 0.0};
    // The paired laboratory experiments isolate their selected bodies from
    // every non-gravitational free-flight term. This is important for the
    // plush intercept target: its ordinary damping is intentionally strong,
    // but internal compliance cannot slow its center of mass in an ideal
    // vacuum. Ordinary chamber bodies retain their authored material damping.
    const Vec3 damping = idealVacuum ? Vec3{}
        : body.velocity * (-spec.linearDamping * dynamicMass(body));
    const Vec3 drag = idealVacuum ? Vec3{} : bodyDrag(body);
    body.force += gravityForce + buoyancy + drag + damping;
    body.velocity += body.force * (inverseMass(body) * dt);
    body.angularVelocity += inverseInertiaWorld(spec, body, body.torque) * dt;
    const Real speed = body.velocity.length();
    if (speed > kMaximumLinearSpeed) body.velocity *= kMaximumLinearSpeed / speed;
    const Real angularSpeed = body.angularVelocity.length();
    if (angularSpeed > kMaximumAngularSpeed) {
        body.angularVelocity *= kMaximumAngularSpeed / angularSpeed;
    }
    body.orientation = body.orientation.integrate(body.angularVelocity, dt);
    body.force = {};
    body.torque = {};
}

void PhysicsWorld::projectBodyInsideRoom(RigidBody& body) const {
    const std::array<std::tuple<int, Vec3, Real>, 6> planes{
        std::tuple{0, Vec3{1.0, 0.0, 0.0}, 0.0},
        std::tuple{0, Vec3{-1.0, 0.0, 0.0}, kRoomWidth},
        std::tuple{1, Vec3{0.0, 1.0, 0.0}, 0.0},
        std::tuple{1, Vec3{0.0, -1.0, 0.0}, kRoomHeight},
        std::tuple{2, Vec3{0.0, 0.0, 1.0}, 0.0},
        std::tuple{2, Vec3{0.0, 0.0, -1.0}, kRoomLength},
    };
    for (const auto& [axis, normal, boundary] : planes) {
        const Vec3 positive{axis == 0 ? 1.0 : 0.0,
                            axis == 1 ? 1.0 : 0.0,
                            axis == 2 ? 1.0 : 0.0};
        const Real extent = supportExtent(body, positive);
        Real* coordinate = axis == 0 ? &body.position.x
                         : axis == 1 ? &body.position.y : &body.position.z;
        const bool minimum = normal.x > 0.0 || normal.y > 0.0 || normal.z > 0.0;
        const Real limit = minimum ? boundary + extent : boundary - extent;
        const bool outside = minimum ? *coordinate < limit : *coordinate > limit;
        if (outside) *coordinate = limit;
        const Vec3 point = supportPoint(body, -normal);
        const Real inward = velocityAt(body, point).dot(normal);
        if ((outside || std::abs(*coordinate - limit) <= 2.0e-6) && inward < 0.0) {
            const Real invMass = inverseMass(body);
            const Vec3 lever = point - body.position;
            const Vec3 cross = lever.cross(normal);
            const Real angular = inverseInertiaWorld(specs_[body.spec], body, cross)
                .cross(lever).dot(normal);
            const Real denominator = invMass + std::max(0.0, angular);
            if (denominator > 1.0e-12) {
                applyImpulseRaw(specs_[body.spec], body, invMass,
                                normal * (-inward / denominator), point);
            }
        }
    }
    const Real vertical = supportExtent(body, {0.0, 1.0, 0.0});
    if (body.position.y <= vertical + 2.0e-5) body.grounded = true;
}

void PhysicsWorld::advanceBodySwept(RigidBody& body, Real dt) {
    if (body.asleep || body.stuckSurface || body.stuckTo >= 0) return;
    // Authored reset stacks begin with known support. Preserve that hint while
    // the untouched stack settles; any player/tuning disturbance clears
    // pristine and returns to contact-derived grounding every tick.
    if (!body.pristine) body.grounded = false;
    projectBodyInsideRoom(body);
    const Vec3 positiveX{1.0, 0.0, 0.0};
    const Vec3 positiveY{0.0, 1.0, 0.0};
    const Vec3 positiveZ{0.0, 0.0, 1.0};
    const Real extentX = supportExtent(body, positiveX);
    const Real extentY = supportExtent(body, positiveY);
    const Real extentZ = supportExtent(body, positiveZ);
    Real remaining = dt;

    const auto roomImpact = [&](const Vec3& normal, const Vec3& contact) {
        const BodySpec& spec = specs_[body.spec];
        const Vec3 relative = velocityAt(body, contact);
        const Real normalSpeed = relative.dot(normal);
        if (normalSpeed >= 0.0) return Real{0.0};
        const Real restitution = -normalSpeed >= kRestitutionThreshold
            ? spec.restitution : 0.0;
        const Vec3 lever = contact - body.position;
        const Vec3 leverCrossNormal = lever.cross(normal);
        const Real angular = inverseInertiaWorld(spec, body, leverCrossNormal)
            .cross(lever).dot(normal);
        const Real invMass = inverseMass(body);
        const Real denominator = invMass + std::max(0.0, angular);
        if (denominator <= 1.0e-12) return Real{0.0};
        const Real magnitude = -(1.0 + restitution) * normalSpeed / denominator;
        applyImpulseRaw(spec, body, invMass, normal * magnitude, contact);
        if (-normalSpeed > kWakeSpeed) wake(body, !body.pristine);

        const Vec3 postVelocity = velocityAt(body, contact);
        const Vec3 tangentVelocity = postVelocity - normal * postVelocity.dot(normal);
        const Real tangentSpeed = tangentVelocity.length();
        if (tangentSpeed > 1.0e-8) {
            const Vec3 tangent = tangentVelocity / tangentSpeed;
            const Vec3 leverCrossTangent = lever.cross(tangent);
            const Real tangentAngular = inverseInertiaWorld(
                spec, body, leverCrossTangent).cross(lever).dot(tangent);
            const Real tangentDenominator = invMass + std::max(0.0, tangentAngular);
            const Real desired = tangentSpeed / std::max(tangentDenominator, 1.0e-9);
            const Real frictionLimit = effectiveRoomFriction(body) * magnitude;
            applyImpulseRaw(spec, body, invMass,
                            tangent * (-std::min(desired, frictionLimit)), contact);
        }
        body.lastImpulse = std::max(body.lastImpulse, magnitude);
        if (body.impactCooldown <= 0.0 && -normalSpeed > kImpactSoundSpeed) {
            if (impacts_.size() < kMaximumFrameImpactEvents) {
                impacts_.push_back({spec.soundFamily, body.position, magnitude,
                                    -normalSpeed, dynamicMass(body)});
            }
            body.impactCooldown = 0.075;
        }
        if (spec.adhesionStrength > 0.0 && -normalSpeed > kAdhesionSpeed
            && !body.held) {
            body.stuckSurface = true;
            body.stuckNormal = normal;
            body.velocity = {};
            body.angularVelocity = {};
            body.asleep = true;
            body.pristine = false;
            addMessage((body.instanceLabel.empty() ? spec.name : body.instanceLabel)
                       + " splatted and adhered to the room surface.");
        }
        return magnitude;
    };

    for (int bounce = 0; bounce < kMaximumCcdBounces && remaining > 1.0e-8;
         ++bounce) {
        Real earliest = remaining + 1.0;
        Vec3 hitNormal{};
        bool found = false;
        bool floor = false;
        struct PlaneCandidate { Real velocity; Real displacement; Vec3 normal; bool floor; };
        const std::array<PlaneCandidate, 6> planes{
            PlaneCandidate{body.velocity.x, extentX - body.position.x, positiveX, false},
            PlaneCandidate{body.velocity.x, kRoomWidth - extentX - body.position.x, -positiveX, false},
            PlaneCandidate{body.velocity.y, extentY - body.position.y, positiveY, true},
            PlaneCandidate{body.velocity.y, kRoomHeight - extentY - body.position.y, -positiveY, false},
            PlaneCandidate{body.velocity.z, extentZ - body.position.z, positiveZ, false},
            PlaneCandidate{body.velocity.z, kRoomLength - extentZ - body.position.z, -positiveZ, false},
        };
        for (const auto& plane : planes) {
            const Real toward = body.velocity.dot(plane.normal);
            if (toward >= -1.0e-12 || std::abs(plane.velocity) <= 1.0e-12) continue;
            const Real collisionTime = plane.displacement / plane.velocity;
            if (collisionTime >= -1.0e-9 && collisionTime <= remaining
                && collisionTime < earliest) {
                earliest = std::max(0.0, collisionTime);
                hitNormal = plane.normal;
                found = true;
                floor = plane.floor;
            }
        }
        if (!found) {
            body.position += body.velocity * remaining;
            remaining = 0.0;
            break;
        }
        body.position += body.velocity * earliest;
        remaining -= earliest;
        roomImpact(hitNormal, supportPoint(body, -hitNormal));
        if (floor) body.grounded = true;
        body.position += hitNormal * 1.0e-6;
        remaining = std::max(0.0, remaining - 1.0e-8);
    }
    if (remaining > 1.0e-8) body.position += body.velocity * remaining;
    projectBodyInsideRoom(body);
}

void PhysicsWorld::projectPlayerInsideRoom() noexcept {
    player_.position.x = clamp(player_.position.x, kPlayerRadius,
                               kRoomWidth - kPlayerRadius);
    player_.position.z = clamp(player_.position.z, kPlayerRadius,
                               kRoomLength - kPlayerRadius);
    player_.position.y = clamp(player_.position.y, 0.0,
                               kRoomHeight - kPlayerHeight);
    if (player_.position.x <= kPlayerRadius && player_.velocity.x < 0.0)
        player_.velocity.x = 0.0;
    if (player_.position.x >= kRoomWidth - kPlayerRadius && player_.velocity.x > 0.0)
        player_.velocity.x = 0.0;
    if (player_.position.z <= kPlayerRadius && player_.velocity.z < 0.0)
        player_.velocity.z = 0.0;
    if (player_.position.z >= kRoomLength - kPlayerRadius && player_.velocity.z > 0.0)
        player_.velocity.z = 0.0;
    if (player_.position.y <= 0.0 && player_.velocity.y < 0.0)
        player_.velocity.y = 0.0;
    if (player_.position.y >= kRoomHeight - kPlayerHeight && player_.velocity.y > 0.0)
        player_.velocity.y = 0.0;
}

bool PhysicsWorld::sweptBoundingSpheres(const RigidBody& a,
                                        const RigidBody& b) const noexcept {
    const Vec3 start = b.previousPosition - a.previousPosition;
    const Vec3 relativeMotion = (b.position - b.previousPosition)
                              - (a.position - a.previousPosition);
    const Real motion2 = relativeMotion.lengthSquared();
    const Real fraction = motion2 > 1.0e-14
        ? clamp(-start.dot(relativeMotion) / motion2, 0.0, 1.0) : 0.0;
    const Vec3 closest = start + relativeMotion * fraction;
    const Real radius = a.cachedBoundingRadius + b.cachedBoundingRadius
                      + kBroadphaseSkin;
    return closest.lengthSquared() <= radius * radius;
}

void PhysicsWorld::buildSpatialBroadphasePairs(
    std::span<const std::uint16_t> active) {
    struct CellCoordinate {
        int x{};
        int y{};
        int z{};
        auto operator<=>(const CellCoordinate&) const = default;
    };
    struct CellEntry {
        std::uint32_t key{};
        std::uint16_t body{};
    };
    struct CellRange {
        std::uint32_t key{};
        std::size_t begin{};
        std::size_t end{};
    };
    struct SpatialScratch {
        std::vector<std::uint8_t> activeMask;
        std::vector<std::uint8_t> globalMask;
        std::vector<CellEntry> entries;
        std::vector<CellRange> ranges;
        std::vector<std::uint16_t> globalBodies;
    };
    static thread_local SpatialScratch scratch;

    constexpr int cellsX = 400;
    constexpr int cellsY = 40;
    constexpr int cellsZ = 400;
    static_assert(kSpatialBroadphaseCellSize == 0.25);
    static_assert(static_cast<std::uint64_t>(cellsX) * cellsY * cellsZ
                  < std::numeric_limits<std::uint32_t>::max());

    const auto coordinateFor = [](const Vec3& position) {
        return CellCoordinate{
            clamp(static_cast<int>(std::floor(
                      position.x / kSpatialBroadphaseCellSize)), 0, cellsX - 1),
            clamp(static_cast<int>(std::floor(
                      position.y / kSpatialBroadphaseCellSize)), 0, cellsY - 1),
            clamp(static_cast<int>(std::floor(
                      position.z / kSpatialBroadphaseCellSize)), 0, cellsZ - 1),
        };
    };
    const auto cellKey = [](const CellCoordinate& cell) {
        return static_cast<std::uint32_t>(
            cell.x + cellsX * (cell.y + cellsY * cell.z));
    };

    const std::size_t bodyCount = bodies_.size();
    scratch.activeMask.resize(bodyCount);
    scratch.globalMask.resize(bodyCount);
    std::fill(scratch.activeMask.begin(), scratch.activeMask.end(), 0U);
    std::fill(scratch.globalMask.begin(), scratch.globalMask.end(), 0U);
    for (const std::uint16_t index : active) scratch.activeMask[index] = 1U;

    scratch.entries.clear();
    scratch.ranges.clear();
    scratch.globalBodies.clear();
    scratch.entries.reserve(bodyCount);
    scratch.ranges.reserve(bodyCount);
    scratch.globalBodies.reserve(64);

    // Compact bodies that remain in one cell use a 27-cell neighborhood.
    // Large bodies and any cell-crossing sweep take a bounded global path.
    // The radius threshold makes the center-cell neighborhood collision-safe.
    constexpr Real maximumLocalRadius = kSpatialBroadphaseCellSize * 0.48;
    for (std::size_t index = 0; index < bodyCount; ++index) {
        const RigidBody& body = bodies_[index];
        const CellCoordinate current = coordinateFor(body.position);
        const CellCoordinate previous = coordinateFor(body.previousPosition);
        const bool global = body.cachedBoundingRadius + kBroadphaseSkin * 0.5
                                > maximumLocalRadius
                         || current != previous;
        if (global) {
            scratch.globalMask[index] = 1U;
            scratch.globalBodies.push_back(static_cast<std::uint16_t>(index));
        } else {
            scratch.entries.push_back(
                {cellKey(current), static_cast<std::uint16_t>(index)});
        }
    }
    std::sort(scratch.entries.begin(), scratch.entries.end(),
        [](const CellEntry& a, const CellEntry& b) {
            return a.key != b.key ? a.key < b.key : a.body < b.body;
        });
    for (std::size_t begin = 0; begin < scratch.entries.size();) {
        std::size_t end = begin + 1U;
        while (end < scratch.entries.size()
               && scratch.entries[end].key == scratch.entries[begin].key) {
            ++end;
        }
        scratch.ranges.push_back({scratch.entries[begin].key, begin, end});
        begin = end;
    }

    const std::size_t workChunks = std::min<std::size_t>(
        8U, std::max<std::size_t>(1U, active.size()));
    threadPairs_.resize(workChunks);
    // Capture immutable pointers to this main thread's TLS storage so worker
    // threads never resolve their own empty scratch instances.
    const CellEntry* const sharedEntries = scratch.entries.data();
    const CellRange* const sharedRanges = scratch.ranges.data();
    const std::size_t sharedRangeCount = scratch.ranges.size();
    const std::uint8_t* const sharedActiveMask = scratch.activeMask.data();
    const std::uint8_t* const sharedGlobalMask = scratch.globalMask.data();
    const std::uint16_t* const sharedGlobals = scratch.globalBodies.data();
    const std::size_t sharedGlobalCount = scratch.globalBodies.size();
    const auto enumerateActiveBodies = [&](std::size_t chunkBegin,
                                           std::size_t chunkEnd) {
        for (std::size_t chunk = chunkBegin; chunk < chunkEnd; ++chunk) {
            auto& output = threadPairs_[chunk].pairs;
            output.clear();
            const std::size_t activeBegin = active.size() * chunk / workChunks;
            const std::size_t activeEnd = active.size() * (chunk + 1U) / workChunks;
            output.reserve((activeEnd - activeBegin) * 8U);
            const auto appendCandidate = [&](std::uint16_t first,
                                             std::uint16_t second) {
                if (first == second
                    || (sharedActiveMask[second] != 0U && first > second)
                    || !sweptBoundingSpheres(bodies_[first], bodies_[second])) {
                    return;
                }
                output.push_back(first < second
                    ? Pair{first, second} : Pair{second, first});
            };
            for (std::size_t activeSlot = activeBegin;
                 activeSlot < activeEnd; ++activeSlot) {
                const std::uint16_t first = active[activeSlot];
                if (sharedGlobalMask[first] != 0U) {
                    for (std::size_t second = 0; second < bodyCount; ++second) {
                        appendCandidate(first, static_cast<std::uint16_t>(second));
                    }
                    continue;
                }
                const CellCoordinate center = coordinateFor(bodies_[first].position);
                for (int dz = -1; dz <= 1; ++dz) {
                    const int z = center.z + dz;
                    if (z < 0 || z >= cellsZ) continue;
                    for (int dy = -1; dy <= 1; ++dy) {
                        const int y = center.y + dy;
                        if (y < 0 || y >= cellsY) continue;
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int x = center.x + dx;
                            if (x < 0 || x >= cellsX) continue;
                            const std::uint32_t key = cellKey({x, y, z});
                            const auto range = std::lower_bound(
                                sharedRanges,
                                sharedRanges + sharedRangeCount, key,
                                [](const CellRange& entry, std::uint32_t wanted) {
                                    return entry.key < wanted;
                                });
                            if (range == sharedRanges + sharedRangeCount
                                || range->key != key) continue;
                            for (std::size_t slot = range->begin;
                                 slot < range->end; ++slot) {
                                appendCandidate(first, sharedEntries[slot].body);
                            }
                        }
                    }
                }
                for (std::size_t globalSlot = 0;
                     globalSlot < sharedGlobalCount; ++globalSlot) {
                    appendCandidate(first, sharedGlobals[globalSlot]);
                }
            }
        }
    };
    if (workChunks > 1U) {
        executor_.parallelFor(workChunks, 1U, enumerateActiveBodies);
    } else {
        enumerateActiveBodies(0U, workChunks);
    }
    for (const auto& buffer : threadPairs_) {
        broadphasePairs_.insert(broadphasePairs_.end(),
                                buffer.pairs.begin(), buffer.pairs.end());
    }
    std::sort(broadphasePairs_.begin(), broadphasePairs_.end());
    broadphasePairs_.erase(std::unique(broadphasePairs_.begin(),
                                      broadphasePairs_.end()),
                           broadphasePairs_.end());
}

void PhysicsWorld::buildBroadphasePairs(std::span<const std::uint16_t> active) {
    const auto started = Clock::now();
    broadphasePairs_.clear();
    if (active.empty() || bodies_.empty()) {
        broadphaseMilliseconds_ = 0.0;
        return;
    }
    const std::size_t bodyCount = bodies_.size();
    const std::size_t activeCount = active.size();

    if (activeCount >= kSpatialBroadphaseThreshold) {
        buildSpatialBroadphasePairs(active);
    } else {

    // Reuse the dense scratch buffers across 120 Hz ticks. This removes the
    // steady stream of small heap allocations from the idle room while keeping
    // each calling thread independent for test worlds and future simulations.
    struct Scratch {
        AlignedRealVector x, y, z;
        AlignedRealVector oldX, oldY, oldZ;
        AlignedRealVector radius;
        std::vector<std::uint8_t> activeMask;
        std::vector<std::uint8_t> matrix;
        std::vector<std::uint8_t> bullet;
        std::vector<std::uint8_t> activeCourse;
    };
    static thread_local Scratch scratch;
    scratch.x.resize(bodyCount);
    scratch.y.resize(bodyCount);
    scratch.z.resize(bodyCount);
    scratch.oldX.resize(bodyCount);
    scratch.oldY.resize(bodyCount);
    scratch.oldZ.resize(bodyCount);
    scratch.radius.resize(bodyCount);
    scratch.activeMask.resize(bodyCount);
    std::fill(scratch.activeMask.begin(), scratch.activeMask.end(), 0U);
    auto& x = scratch.x;
    auto& y = scratch.y;
    auto& z = scratch.z;
    auto& oldX = scratch.oldX;
    auto& oldY = scratch.oldY;
    auto& oldZ = scratch.oldZ;
    auto& radius = scratch.radius;
    auto& activeMask = scratch.activeMask;

    // Aligned SoA storage lets GCC/Clang emit paired AdvSIMD double arithmetic
    // on Cortex-A76. Pair materialization remains ordered so push_back branches
    // cannot inhibit vectorization or make results schedule-dependent.
    for (std::size_t index = 0; index < bodyCount; ++index) {
        const RigidBody& body = bodies_[index];
        x[index] = body.position.x;
        y[index] = body.position.y;
        z[index] = body.position.z;
        oldX[index] = body.previousPosition.x;
        oldY[index] = body.previousPosition.y;
        oldZ[index] = body.previousPosition.z;
        radius[index] = body.cachedBoundingRadius + kBroadphaseSkin * 0.5;
    }
    for (const auto index : active) activeMask[index] = 1;

    scratch.matrix.resize(activeCount * bodyCount);
    auto& matrix = scratch.matrix;
    const auto fillRows = [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            const std::size_t first = active[row];
            std::uint8_t* output = matrix.data() + row * bodyCount;
            const Real firstDx = x[first] - oldX[first];
            const Real firstDy = y[first] - oldY[first];
            const Real firstDz = z[first] - oldZ[first];
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
            for (std::size_t second = 0; second < bodyCount; ++second) {
                const Real sx = oldX[second] - oldX[first];
                const Real sy = oldY[second] - oldY[first];
                const Real sz = oldZ[second] - oldZ[first];
                const Real mx = (x[second] - oldX[second]) - firstDx;
                const Real my = (y[second] - oldY[second]) - firstDy;
                const Real mz = (z[second] - oldZ[second]) - firstDz;
                const Real motion2 = mx * mx + my * my + mz * mz;
                const Real projection = -(sx * mx + sy * my + sz * mz);
                const Real fraction = motion2 > 1.0e-14
                    ? clamp(projection / motion2, 0.0, 1.0) : 0.0;
                const Real cx = sx + mx * fraction;
                const Real cy = sy + my * fraction;
                const Real cz = sz + mz * fraction;
                const Real combined = radius[first] + radius[second];
                const bool unique = first != second
                    && (!activeMask[second] || first < second);
                output[second] = static_cast<std::uint8_t>(
                    unique && cx * cx + cy * cy + cz * cz <= combined * combined);
            }
        }
    };
    const std::size_t broadphaseTests = activeCount * bodyCount;
    const bool useParallelBroadphase = cpuBackend_ == "maximum"
        || (cpuBackend_ == "auto"
            && broadphaseTests >= kParallelBroadphaseTests);
    if (!useParallelBroadphase) {
        fillRows(0, activeCount);
    } else {
        executor_.parallelFor(
            activeCount, cpuBackend_ == "maximum" ? 1U : 4U, fillRows);
    }

    for (std::size_t row = 0; row < activeCount; ++row) {
        const std::uint16_t activeIndex = active[row];
        const std::uint8_t* flags = matrix.data() + row * bodyCount;
        for (std::size_t other = 0; other < bodyCount; ++other) {
            if (!flags[other]) continue;
            const auto otherIndex = static_cast<std::uint16_t>(other);
            broadphasePairs_.push_back(activeIndex < otherIndex
                ? Pair{activeIndex, otherIndex} : Pair{otherIndex, activeIndex});
        }
    }
    std::sort(broadphasePairs_.begin(), broadphasePairs_.end());
    broadphasePairs_.erase(std::unique(broadphasePairs_.begin(),
                                      broadphasePairs_.end()),
                           broadphasePairs_.end());
    }

    // A fast pre-impact trajectory is allowed to resolve only its earliest
    // two plausible contacts. Later contacts must be generated from the
    // changed trajectory on the following 120 Hz tick.
    static thread_local std::vector<std::uint8_t> bullet;
    bullet.resize(bodyCount);
    for (std::size_t index = 0; index < bodyCount; ++index) {
        const RigidBody& body = bodies_[index];
        const BodySpec& spec = specs_[body.spec];
        const Real minimumDimension = spec.shape == Shape::Sphere
            ? 0.0 : std::min({spec.dimensions.x, spec.dimensions.y,
                              spec.dimensions.z});
        const Real displacement = (body.position - body.previousPosition).length();
        bullet[index] = static_cast<std::uint8_t>(
            !body.asleep && displacement > std::max(0.04, minimumDimension * 0.45));
    }
    if (std::any_of(bullet.begin(), bullet.end(), [](auto value) { return value != 0; })) {
        std::vector<Pair> retained;
        retained.reserve(broadphasePairs_.size());
        std::unordered_set<std::uint32_t> selected;
        for (std::size_t index = 0; index < bodyCount; ++index) {
            if (!bullet[index]) continue;
            std::vector<std::pair<Real, Pair>> ranked;
            for (const Pair pair : broadphasePairs_) {
                if (pair.first != index && pair.second != index) continue;
                const RigidBody& a = bodies_[pair.first];
                const RigidBody& b = bodies_[pair.second];
                ranked.emplace_back(closestSweepFraction(a, b), pair);
            }
            std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a,
                                                               const auto& b) {
                return a.first != b.first ? a.first < b.first : a.second < b.second;
            });
            for (int admitted = 0;
                 admitted < kMaximumBulletContacts
                 && admitted < static_cast<int>(ranked.size()); ++admitted) {
                selected.insert(pairCode(ranked[static_cast<std::size_t>(admitted)].second));
            }
        }
        for (const Pair pair : broadphasePairs_) {
            if ((!bullet[pair.first] && !bullet[pair.second])
                || selected.contains(pairCode(pair))) {
                retained.push_back(pair);
            }
        }
        broadphasePairs_.swap(retained);
    }

    // Bound dense course-to-course work while retaining all interactions with
    // non-course objects.
    static thread_local std::vector<std::uint8_t> activeCourse;
    activeCourse.resize(bodyCount);
    bool anyActiveCourse = false;
    for (std::size_t index = 0; index < bodyCount; ++index) {
        const BodySpec& spec = specs_[bodies_[index].spec];
        activeCourse[index] = static_cast<std::uint8_t>(
            !bodies_[index].asleep && isCourseStock(spec));
        anyActiveCourse = anyActiveCourse || activeCourse[index];
    }
    if (anyActiveCourse) {
        using RankedPair = std::pair<Real, Pair>;
        static thread_local std::vector<std::vector<RankedPair>> courseNeighbors;
        courseNeighbors.resize(bodyCount);
        for (auto& neighbors : courseNeighbors) neighbors.clear();
        for (std::size_t index = 0; index < bodyCount; ++index) {
            if (activeCourse[index]) courseNeighbors[index].reserve(12);
        }
        for (const Pair pair : broadphasePairs_) {
            const BodySpec& firstSpec = specs_[bodies_[pair.first].spec];
            const BodySpec& secondSpec = specs_[bodies_[pair.second].spec];
            if (!isCourseStock(firstSpec) || !isCourseStock(secondSpec)) continue;
            const Real distance2 =
                (bodies_[pair.second].position - bodies_[pair.first].position)
                    .lengthSquared();
            if (activeCourse[pair.first]) {
                courseNeighbors[pair.first].emplace_back(distance2, pair);
            }
            if (activeCourse[pair.second]) {
                courseNeighbors[pair.second].emplace_back(distance2, pair);
            }
        }
        std::unordered_set<std::uint32_t> selectedCoursePairs;
        selectedCoursePairs.reserve(bodyCount * 2U);
        for (std::size_t index = 0; index < bodyCount; ++index) {
            if (!activeCourse[index]) continue;
            auto& ranked = courseNeighbors[index];
            std::stable_sort(ranked.begin(), ranked.end());
            const std::size_t limit = std::min<std::size_t>(
                kMaximumBrickNeighbors, ranked.size());
            for (std::size_t candidate = 0; candidate < limit; ++candidate) {
                selectedCoursePairs.insert(pairCode(ranked[candidate].second));
            }
        }
        broadphasePairs_.erase(
            std::remove_if(broadphasePairs_.begin(), broadphasePairs_.end(),
                [&](const Pair pair) {
                    const bool bothCourse =
                        isCourseStock(specs_[bodies_[pair.first].spec])
                        && isCourseStock(specs_[bodies_[pair.second].spec]);
                    return bothCourse
                        && !selectedCoursePairs.contains(pairCode(pair));
                }),
            broadphasePairs_.end());
    }
    std::sort(broadphasePairs_.begin(), broadphasePairs_.end());
    broadphaseMilliseconds_ = std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
}

std::optional<PhysicsWorld::Contact> PhysicsWorld::sphereSphereContact(
    std::uint16_t first, std::uint16_t second) const {
    const RigidBody& a = bodies_[first];
    const RigidBody& b = bodies_[second];
    const Real radius = specs_[a.spec].radius() + specs_[b.spec].radius();
    Vec3 delta = b.position - a.position;
    Real distance2 = delta.lengthSquared();
    Real toi = 1.0;
    if (distance2 > radius * radius) {
        const Vec3 start = b.previousPosition - a.previousPosition;
        const Vec3 motion = (b.position - b.previousPosition)
                          - (a.position - a.previousPosition);
        const Real c = start.lengthSquared() - radius * radius;
        const Real aa = motion.lengthSquared();
        if (aa <= 1.0e-14) return std::nullopt;
        const Real bb = 2.0 * start.dot(motion);
        const Real discriminant = bb * bb - 4.0 * aa * c;
        if (discriminant < 0.0) return std::nullopt;
        toi = (-bb - std::sqrt(discriminant)) / (2.0 * aa);
        if (toi < 0.0 || toi > 1.0) return std::nullopt;
        delta = start + motion * toi;
        distance2 = delta.lengthSquared();
    }
    const Real distance = std::sqrt(std::max(distance2, 1.0e-16));
    const Vec3 normal = distance > 1.0e-8
        ? delta / distance : Vec3{1.0, 0.0, 0.0};
    const Vec3 aAt = a.previousPosition
                   + (a.position - a.previousPosition) * toi;
    return Contact{first, second, normal,
                   aAt + normal * specs_[a.spec].radius(),
                   std::max(0.0, radius - distance), toi};
}

std::optional<PhysicsWorld::Contact> PhysicsWorld::sphereBoxContact(
    std::uint16_t sphereIndex, std::uint16_t boxIndex, bool flip) const {
    const RigidBody& sphere = bodies_[sphereIndex];
    const RigidBody& box = bodies_[boxIndex];
    const BodySpec& sphereSpec = specs_[sphere.spec];
    const BodySpec& boxSpec = specs_[box.spec];
    const Quat inverse = box.orientation.conjugate();
    const Vec3 half = boxSpec.halfExtents();
    const Real sphereRadius = sphereSpec.radius();

    auto contactAt = [&](Real fraction) -> std::optional<Contact> {
        const Vec3 spherePosition = sphere.previousPosition
            + (sphere.position - sphere.previousPosition) * fraction;
        const Vec3 boxPosition = box.previousPosition
            + (box.position - box.previousPosition) * fraction;
        const Vec3 localCenter = inverse.rotate(spherePosition - boxPosition);
        const Vec3 closest{
            clamp(localCenter.x, -half.x, half.x),
            clamp(localCenter.y, -half.y, half.y),
            clamp(localCenter.z, -half.z, half.z),
        };
        const Vec3 towardBox = closest - localCenter;
        const Real distance2 = towardBox.lengthSquared();
        if (distance2 > sphereRadius * sphereRadius + 1.0e-10) return std::nullopt;
        Vec3 normalLocal;
        Real penetration = 0.0;
        Vec3 contactLocal = closest;
        if (distance2 > 1.0e-14) {
            const Real distance = std::sqrt(distance2);
            normalLocal = towardBox / distance;
            penetration = std::max(0.0, sphereRadius - distance);
        } else {
            const std::array<std::pair<Real, Vec3>, 3> faces{
                std::pair{half.x - std::abs(localCenter.x),
                          Vec3{localCenter.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0}},
                std::pair{half.y - std::abs(localCenter.y),
                          Vec3{0.0, localCenter.y >= 0.0 ? 1.0 : -1.0, 0.0}},
                std::pair{half.z - std::abs(localCenter.z),
                          Vec3{0.0, 0.0, localCenter.z >= 0.0 ? 1.0 : -1.0}},
            };
            const auto nearest = std::min_element(faces.begin(), faces.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            normalLocal = -nearest->second;
            penetration = sphereRadius + nearest->first;
            contactLocal = localCenter + nearest->second * nearest->first;
        }
        Vec3 normal = box.orientation.rotate(normalLocal)
            .normalized({1.0, 0.0, 0.0});
        const Vec3 point = boxPosition + box.orientation.rotate(contactLocal);
        if (flip) {
            normal = -normal;
            return Contact{boxIndex, sphereIndex, normal, point, penetration, fraction};
        }
        return Contact{sphereIndex, boxIndex, normal, point, penetration, fraction};
    };

    if (auto current = contactAt(1.0)) return current;
    const Vec3 start = inverse.rotate(sphere.previousPosition - box.previousPosition);
    const Vec3 end = inverse.rotate(sphere.position - box.position);
    const Vec3 motion = end - start;
    const Vec3 lower{-half.x - sphereRadius, -half.y - sphereRadius,
                     -half.z - sphereRadius};
    const Vec3 upper{half.x + sphereRadius, half.y + sphereRadius,
                     half.z + sphereRadius};
    Real entry = 0.0;
    Real exit = 1.0;
    const std::array<std::tuple<Real, Real, Real, Real>, 3> slabs{
        std::tuple{start.x, motion.x, lower.x, upper.x},
        std::tuple{start.y, motion.y, lower.y, upper.y},
        std::tuple{start.z, motion.z, lower.z, upper.z},
    };
    for (const auto& [origin, velocity, low, high] : slabs) {
        if (std::abs(velocity) <= 1.0e-14) {
            if (origin < low || origin > high) return std::nullopt;
            continue;
        }
        Real first = (low - origin) / velocity;
        Real second = (high - origin) / velocity;
        if (first > second) std::swap(first, second);
        entry = std::max(entry, first);
        exit = std::min(exit, second);
        if (entry > exit) return std::nullopt;
    }
    if (entry < 0.0 || entry > 1.0) return std::nullopt;
    // Conservative advancement rejects the square expanded-box corner region.
    const Real motionLength = motion.length();
    if (motionLength <= 1.0e-14) return std::nullopt;
    Real fraction = entry;
    for (int iteration = 0; iteration < 24; ++iteration) {
        const Vec3 point = start + motion * fraction;
        const Vec3 closest{
            clamp(point.x, -half.x, half.x),
            clamp(point.y, -half.y, half.y),
            clamp(point.z, -half.z, half.z),
        };
        const Real separation = (point - closest).length() - sphereRadius;
        if (separation <= 1.0e-7) {
            if (auto swept = contactAt(fraction)) return swept;
            // At a numerically exterior TOI, construct a zero-penetration
            // contact from the rounded-box closest point.
            const Vec3 localDelta = closest - point;
            Vec3 normal = box.orientation.rotate(
                localDelta.normalized({1.0, 0.0, 0.0}));
            const Vec3 boxPosition = box.previousPosition
                + (box.position - box.previousPosition) * fraction;
            const Vec3 worldPoint = boxPosition + box.orientation.rotate(closest);
            if (flip) return Contact{boxIndex, sphereIndex, -normal,
                                     worldPoint, 0.0, fraction};
            return Contact{sphereIndex, boxIndex, normal,
                           worldPoint, 0.0, fraction};
        }
        fraction += std::max(1.0e-7, separation / motionLength);
        if (fraction > exit + 1.0e-9 || fraction > 1.0) return std::nullopt;
    }
    return std::nullopt;
}

std::optional<PhysicsWorld::Contact> PhysicsWorld::boxBoxContact(
    std::uint16_t first, std::uint16_t second) const {
    const RigidBody& a = bodies_[first];
    const RigidBody& b = bodies_[second];
    const auto axesA = boxAxes(a);
    const auto axesB = boxAxes(b);
    const Vec3 halfA = specs_[a.spec].halfExtents();
    const Vec3 halfB = specs_[b.spec].halfExtents();
    std::vector<Vec3> axes;
    axes.reserve(15);
    axes.insert(axes.end(), axesA.begin(), axesA.end());
    axes.insert(axes.end(), axesB.begin(), axesB.end());
    for (const Vec3& axisA : axesA) {
        for (const Vec3& axisB : axesB) {
            const Vec3 cross = axisA.cross(axisB);
            if (cross.lengthSquared() > 1.0e-12) axes.push_back(cross.normalized());
        }
    }

    const Vec3 currentDelta = b.position - a.position;
    Real minimumOverlap = std::numeric_limits<Real>::infinity();
    Vec3 minimumAxis{1.0, 0.0, 0.0};
    bool currentOverlap = true;
    for (const Vec3& axis : axes) {
        const Real radiusA = projectedRadius(axesA, halfA, axis);
        const Real radiusB = projectedRadius(axesB, halfB, axis);
        const Real signedDistance = currentDelta.dot(axis);
        const Real overlap = radiusA + radiusB - std::abs(signedDistance);
        if (overlap <= 0.0) {
            currentOverlap = false;
            break;
        }
        if (overlap < minimumOverlap) {
            minimumOverlap = overlap;
            minimumAxis = signedDistance >= 0.0 ? axis : -axis;
        }
    }
    Real toi = 1.0;
    if (!currentOverlap) {
        const Vec3 start = b.previousPosition - a.previousPosition;
        const Vec3 motion = currentDelta - start;
        Real entry = 0.0;
        Real exit = 1.0;
        minimumAxis = {1.0, 0.0, 0.0};
        for (const Vec3& axis : axes) {
            const Real radius = projectedRadius(axesA, halfA, axis)
                              + projectedRadius(axesB, halfB, axis);
            const Real origin = start.dot(axis);
            const Real speed = motion.dot(axis);
            if (std::abs(speed) <= 1.0e-14) {
                if (std::abs(origin) > radius) return std::nullopt;
                continue;
            }
            Real firstTime = (-radius - origin) / speed;
            Real secondTime = (radius - origin) / speed;
            if (firstTime > secondTime) std::swap(firstTime, secondTime);
            if (firstTime > entry) {
                entry = firstTime;
                const Real signedAtEntry = origin + speed * entry;
                minimumAxis = signedAtEntry >= 0.0 ? axis : -axis;
            }
            exit = std::min(exit, secondTime);
            if (entry > exit) return std::nullopt;
        }
        if (exit < 0.0 || entry > 1.0) return std::nullopt;
        toi = clamp(entry, 0.0, 1.0);
        minimumOverlap = 0.0;
    }
    const Vec3 aPosition = a.previousPosition + (a.position - a.previousPosition) * toi;
    const Vec3 bPosition = b.previousPosition + (b.position - b.previousPosition) * toi;
    auto supportAt = [&](const RigidBody& body, const BodySpec& spec,
                         const Vec3& position, const Vec3& direction) {
        const Vec3 half = spec.halfExtents();
        const Vec3 localDirection = body.orientation.conjugate().rotate(direction);
        const Vec3 local{
            localDirection.x > 1.0e-10 ? half.x : localDirection.x < -1.0e-10 ? -half.x : 0.0,
            localDirection.y > 1.0e-10 ? half.y : localDirection.y < -1.0e-10 ? -half.y : 0.0,
            localDirection.z > 1.0e-10 ? half.z : localDirection.z < -1.0e-10 ? -half.z : 0.0,
        };
        return position + body.orientation.rotate(local);
    };
    const Vec3 pointA = supportAt(a, specs_[a.spec], aPosition, minimumAxis);
    const Vec3 pointB = supportAt(b, specs_[b.spec], bPosition, -minimumAxis);
    return Contact{first, second, minimumAxis, (pointA + pointB) * 0.5,
                   std::max(0.0, minimumOverlap), toi};
}

std::optional<PhysicsWorld::Contact> PhysicsWorld::findContact(
    std::uint16_t first, std::uint16_t second) const {
    const Shape firstShape = specs_[bodies_[first].spec].shape;
    const Shape secondShape = specs_[bodies_[second].spec].shape;
    if (firstShape == Shape::Sphere && secondShape == Shape::Sphere)
        return sphereSphereContact(first, second);
    if (firstShape == Shape::Sphere && secondShape == Shape::Box)
        return sphereBoxContact(first, second, false);
    if (firstShape == Shape::Box && secondShape == Shape::Sphere)
        return sphereBoxContact(second, first, true);
    return boxBoxContact(first, second);
}

Real PhysicsWorld::applyBodyImpulse(const Contact& contact,
                                    bool positionCorrection) {
    RigidBody& a = bodies_[contact.first];
    RigidBody& b = bodies_[contact.second];
    if ((a.held && b.held) || a.stuckSurface || b.stuckSurface
        || a.stuckTo >= 0 || b.stuckTo >= 0) {
        return 0.0;
    }
    const BodySpec& specA = specs_[a.spec];
    const BodySpec& specB = specs_[b.spec];
    const Real invA = a.asleep ? 0.0 : inverseMass(a);
    const Real invB = b.asleep ? 0.0 : inverseMass(b);
    if (positionCorrection && contact.penetration > 0.0) {
        const Real total = invA + invB;
        if (total > 1.0e-12) {
            Real factor = 0.62;
            if (specA.shape == Shape::Box && specB.shape == Shape::Box) factor = 0.64;
            else if (specA.shape != specB.shape) factor = 0.66;
            const Vec3 correction = contact.normal
                * (std::max(0.0, contact.penetration - kPositionSlop)
                   * factor / total);
            a.position -= correction * invA;
            b.position += correction * invB;
        }
    }

    const Vec3 ra = contact.point - a.position;
    const Vec3 rb = contact.point - b.position;
    const Vec3 relative = velocityAt(b, contact.point)
                        - velocityAt(a, contact.point);
    const Real normalSpeed = relative.dot(contact.normal);
    if (contact.normal.y > 0.55) b.grounded = true;
    else if (contact.normal.y < -0.55) a.grounded = true;
    if (normalSpeed >= 0.0) return 0.0;
    const Real restitution = -normalSpeed >= kRestitutionThreshold
        ? std::sqrt(specA.restitution * specB.restitution) : 0.0;
    const Vec3 raCross = ra.cross(contact.normal);
    const Vec3 rbCross = rb.cross(contact.normal);
    const Real angularA = a.asleep ? 0.0
        : inverseInertiaWorld(specA, a, raCross).cross(ra).dot(contact.normal);
    const Real angularB = b.asleep ? 0.0
        : inverseInertiaWorld(specB, b, rbCross).cross(rb).dot(contact.normal);
    const Real denominator = invA + invB + std::max(0.0, angularA)
                           + std::max(0.0, angularB);
    if (denominator <= 1.0e-12) return 0.0;
    const Real magnitude = -(1.0 + restitution) * normalSpeed / denominator;
    const Vec3 impulse = contact.normal * magnitude;
    const bool meaningful = -normalSpeed > kWakeSpeed;
    const bool externallyDisturbed = !a.pristine || !b.pristine;

    auto deliver = [&](std::uint16_t index, const Vec3& bodyImpulse,
                       const Vec3& point) {
        RigidBody& body = bodies_[index];
        if (!body.asleep) {
            applyImpulseRaw(specs_[body.spec], body, inverseMass(body),
                            bodyImpulse, point);
            if (meaningful) wake(body, externallyDisturbed);
            return;
        }
        if (!meaningful || bodyImpulse.lengthSquared() <= 1.0e-12) return;
        auto existing = std::find_if(pendingWakes_.begin(), pendingWakes_.end(),
            [&](const PendingWake& wakeEntry) { return wakeEntry.body == index; });
        if (existing == pendingWakes_.end()) {
            pendingWakes_.push_back({index, bodyImpulse, point});
        } else {
            existing->impulse += bodyImpulse;
            existing->point = point;
        }
    };
    deliver(contact.first, -impulse, contact.point);
    deliver(contact.second, impulse, contact.point);

    const Vec3 postRelative = velocityAt(b, contact.point)
                            - velocityAt(a, contact.point);
    const Vec3 tangentVelocity = postRelative
        - contact.normal * postRelative.dot(contact.normal);
    const Real tangentSpeed = tangentVelocity.length();
    if (tangentSpeed > 1.0e-8) {
        const Vec3 tangent = tangentVelocity / tangentSpeed;
        const Vec3 raCrossTangent = ra.cross(tangent);
        const Vec3 rbCrossTangent = rb.cross(tangent);
        const Real tangentAngularA = a.asleep ? 0.0
            : inverseInertiaWorld(specA, a, raCrossTangent)
                .cross(ra).dot(tangent);
        const Real tangentAngularB = b.asleep ? 0.0
            : inverseInertiaWorld(specB, b, rbCrossTangent)
                .cross(rb).dot(tangent);
        const Real tangentDenominator = invA + invB
            + std::max(0.0, tangentAngularA) + std::max(0.0, tangentAngularB);
        const Real desired = tangentSpeed / std::max(tangentDenominator, 1.0e-9);
        const Real coefficient = std::sqrt(specA.friction * specB.friction);
        const Vec3 frictionImpulse = tangent
            * (-std::min(desired, coefficient * magnitude));
        deliver(contact.first, -frictionImpulse, contact.point);
        deliver(contact.second, frictionImpulse, contact.point);
    }

    RigidBody& louder = dynamicMass(a) <= dynamicMass(b) ? a : b;
    const BodySpec& louderSpec = specs_[louder.spec];
    if (louder.impactCooldown <= 0.0 && -normalSpeed > kImpactSoundSpeed) {
        if (impacts_.size() < kMaximumFrameImpactEvents) {
            impacts_.push_back({louderSpec.soundFamily, contact.point, magnitude,
                                -normalSpeed, dynamicMass(louder)});
        }
        louder.impactCooldown = 0.075;
    }
    a.lastImpulse = std::max(a.lastImpulse, magnitude);
    b.lastImpulse = std::max(b.lastImpulse, magnitude);

    if (-normalSpeed > kAdhesionSpeed) {
        const auto attach = [&](std::uint16_t gooIndex, std::uint16_t hostIndex) {
            RigidBody& goo = bodies_[gooIndex];
            RigidBody& host = bodies_[hostIndex];
            const BodySpec& gooSpec = specs_[goo.spec];
            if (gooSpec.adhesionStrength <= 0.0 || goo.held || goo.stuckTo >= 0
                || goo.stuckSurface || gooIndex == hostIndex) return;
            const Real gooMass = dynamicMass(goo);
            const Real hostMass = dynamicMass(host);
            const Real combined = gooMass + hostMass;
            if (combined > 1.0e-12) {
                host.velocity = (host.velocity * hostMass + goo.velocity * gooMass)
                              / combined;
            }
            host.attachedPayloadMass += gooMass;
            goo.stuckTo = hostIndex;
            goo.massCarriedByHost = true;
            goo.stuckLocalPosition = host.orientation.conjugate().rotate(
                goo.position - host.position);
            goo.velocity = velocityAt(host, goo.position);
            goo.angularVelocity = {};
            goo.pristine = false;
            const std::string gooName = goo.instanceLabel.empty()
                ? gooSpec.name : goo.instanceLabel;
            const BodySpec& hostSpec = specs_[host.spec];
            const std::string hostName = host.instanceLabel.empty()
                ? hostSpec.name : host.instanceLabel;
            addMessage(gooName + " adhered to " + hostName + '.');
        };
        if (specA.adhesionStrength > 0.0) {
            attach(contact.first, contact.second);
        } else if (specB.adhesionStrength > 0.0) {
            attach(contact.second, contact.first);
        }
    }
    return magnitude;
}

void PhysicsWorld::resolvePlayerBody(std::uint16_t bodyIndex, Real dt) {
    RigidBody& body = bodies_[bodyIndex];
    if (body.held || body.stuckSurface || body.stuckTo >= 0) return;
    const BodySpec& spec = specs_[body.spec];
    const Real capsuleLow = player_.position.y + kPlayerRadius;
    const Real capsuleHigh = player_.position.y + kPlayerHeight - kPlayerRadius;
    Vec3 normal{};
    Vec3 point{};
    Real penetration = 0.0;
    bool contactFound = false;

    if (spec.shape == Shape::Sphere) {
        const Vec3 closest{player_.position.x,
                           clamp(body.position.y, capsuleLow, capsuleHigh),
                           player_.position.z};
        const Vec3 delta = body.position - closest;
        const Real combined = kPlayerRadius + spec.radius();
        const Real distance2 = delta.lengthSquared();
        if (distance2 < combined * combined) {
            const Real distance = std::sqrt(std::max(distance2, 1.0e-16));
            normal = distance > 1.0e-8
                ? delta / distance : Vec3{1.0, 0.0, 0.0};
            point = body.position - normal * spec.radius();
            penetration = combined - distance;
            contactFound = true;
        }
    } else {
        const Quat inverse = body.orientation.conjugate();
        const Vec3 half = spec.halfExtents();
        Real bestDistance2 = std::numeric_limits<Real>::infinity();
        Vec3 bestDelta{};
        for (int sample = 0; sample < 9; ++sample) {
            const Real fraction = sample / 8.0;
            const Vec3 axisPoint{
                player_.position.x,
                capsuleLow + (capsuleHigh - capsuleLow) * fraction,
                player_.position.z,
            };
            const Vec3 local = inverse.rotate(axisPoint - body.position);
            const Vec3 closestLocal{
                clamp(local.x, -half.x, half.x),
                clamp(local.y, -half.y, half.y),
                clamp(local.z, -half.z, half.z),
            };
            const Vec3 closestWorld = body.position
                                    + body.orientation.rotate(closestLocal);
            const Vec3 delta = closestWorld - axisPoint;
            if (delta.lengthSquared() < bestDistance2) {
                bestDistance2 = delta.lengthSquared();
                bestDelta = delta;
                point = closestWorld;
            }
        }
        if (bestDistance2 < kPlayerRadius * kPlayerRadius) {
            const Real distance = std::sqrt(std::max(bestDistance2, 1.0e-16));
            normal = distance > 1.0e-8 ? bestDelta / distance
                : (body.position - player_.position).horizontal()
                    .normalized({1.0, 0.0, 0.0});
            penetration = kPlayerRadius - distance;
            contactFound = true;
        }
    }

    Real sweptFraction = 1.0;
    if (!contactFound && dt > 0.0) {
        const Vec3 start = body.previousPosition - player_.previousPosition;
        const Vec3 end = body.position - player_.position;
        const Vec3 motion = end - start;
        const Real combined = kPlayerRadius + body.cachedBoundingRadius;
        const Real low = kPlayerRadius;
        const Real high = kPlayerHeight - kPlayerRadius;
        std::vector<Real> candidates;
        const Real aaHorizontal = motion.x * motion.x + motion.z * motion.z;
        const Real bbHorizontal = 2.0 * (start.x * motion.x + start.z * motion.z);
        const Real ccHorizontal = start.x * start.x + start.z * start.z
                                - combined * combined;
        if (aaHorizontal > 1.0e-14) {
            const Real discriminant = bbHorizontal * bbHorizontal
                                    - 4.0 * aaHorizontal * ccHorizontal;
            if (discriminant >= 0.0) {
                const Real root = std::sqrt(discriminant);
                for (const Real fraction : {
                        (-bbHorizontal - root) / (2.0 * aaHorizontal),
                        (-bbHorizontal + root) / (2.0 * aaHorizontal)}) {
                    const Real axial = start.y + motion.y * fraction;
                    if (fraction >= 0.0 && fraction <= 1.0
                        && axial >= low && axial <= high) {
                        candidates.push_back(fraction);
                    }
                }
            }
        }
        for (const Real capY : {low, high}) {
            const Vec3 offset = start - Vec3{0.0, capY, 0.0};
            const Real aa = motion.lengthSquared();
            const Real bb = 2.0 * offset.dot(motion);
            const Real cc = offset.lengthSquared() - combined * combined;
            if (cc <= 0.0) candidates.push_back(0.0);
            else if (aa > 1.0e-14) {
                const Real discriminant = bb * bb - 4.0 * aa * cc;
                if (discriminant >= 0.0) {
                    const Real fraction = (-bb - std::sqrt(discriminant)) / (2.0 * aa);
                    if (fraction >= 0.0 && fraction <= 1.0) candidates.push_back(fraction);
                }
            }
        }
        if (!candidates.empty()) {
            sweptFraction = *std::min_element(candidates.begin(), candidates.end());
            const Vec3 relative = start + motion * sweptFraction;
            const Vec3 closest{0.0, clamp(relative.y, low, high), 0.0};
            normal = (relative - closest).normalized({1.0, 0.0, 0.0});
            body.position = body.previousPosition
                          + (body.position - body.previousPosition) * sweptFraction;
            point = body.position - normal * body.cachedBoundingRadius;
            penetration = 0.0;
            contactFound = true;
        }
    }
    if (!contactFound) return;

    const auto queuePlayerWake = [&](const Vec3& impulse,
                                     const Vec3& wakePoint) {
        auto existing = std::find_if(pendingWakes_.begin(), pendingWakes_.end(),
            [&](const PendingWake& wakeEntry) {
                return wakeEntry.body == bodyIndex;
            });
        if (existing == pendingWakes_.end()) {
            pendingWakes_.push_back({bodyIndex, impulse, wakePoint});
        } else {
            existing->impulse += impulse;
            existing->point = wakePoint;
        }
    };

    // A locked player is an immovable capsule. If it is locked while already
    // overlapping a sleeping body, wake that body so separation has a movable
    // side instead of leaving a permanent static/static overlap.
    if (playerPositionLocked_ && body.asleep && penetration > kPositionSlop) {
        if (isBulkStock(spec)) queuePlayerWake({}, point);
        else wake(body);
    }
    const Real bodyInv = body.asleep ? 0.0 : inverseMass(body);
    const Real playerInv = playerPositionLocked_ ? 0.0 : 1.0 / kPlayerMass;
    const Real totalInv = playerInv + bodyInv;
    if (penetration > 0.0 && totalInv > 1.0e-12) {
        const Vec3 correction = normal
            * (std::max(0.0, penetration - kPositionSlop) * 0.72 / totalInv);
        if (!playerPositionLocked_) {
            player_.position -= correction * playerInv;
        }
        body.position += correction * bodyInv;
    }
    const Vec3 relativeVelocity = velocityAt(body, point) - player_.velocity;
    const Real normalSpeed = relativeVelocity.dot(normal);
    if (normalSpeed >= 0.0) return;
    const Vec3 lever = point - body.position;
    const Vec3 cross = lever.cross(normal);
    const Real angular = body.asleep ? 0.0
        : inverseInertiaWorld(spec, body, cross).cross(lever).dot(normal);
    const Real denominator = playerInv + bodyInv + std::max(0.0, angular);
    const Real restitution = -normalSpeed >= kRestitutionThreshold
        ? std::min(0.18, spec.restitution) : 0.0;
    const Real magnitude = -(1.0 + restitution) * normalSpeed
                         / std::max(denominator, 1.0e-9);
    const Vec3 impulse = normal * magnitude;
    if (body.asleep) queuePlayerWake(impulse, point);
    else {
        applyImpulseRaw(spec, body, bodyInv, impulse, point);
        if (-normalSpeed > kWakeSpeed) wake(body);
    }
    if (!playerPositionLocked_) {
        player_.velocity -= impulse * playerInv;
    }
    if (sweptFraction < 1.0 && dt > 0.0) {
        const Real remaining = dt * (1.0 - sweptFraction);
        if (!playerPositionLocked_) {
            player_.position -= impulse * playerInv * remaining;
        }
        advanceBodySwept(body, remaining);
        if (!playerPositionLocked_) projectPlayerInsideRoom();
    }
}

void PhysicsWorld::updateAttachments() {
    for (auto& body : bodies_) {
        if (body.stuckTo < 0 || body.stuckTo >= static_cast<int>(bodies_.size())) continue;
        const RigidBody& host = bodies_[static_cast<std::size_t>(body.stuckTo)];
        body.previousPosition = body.position;
        body.position = host.position
                      + host.orientation.rotate(body.stuckLocalPosition);
        body.velocity = velocityAt(host, body.position);
        body.angularVelocity = host.angularVelocity;
        body.asleep = host.asleep;
    }
}

void PhysicsWorld::updateWheelbarrowCargo(Real dt) {
    if (wheelbarrowIndex_ < 0
        || wheelbarrowIndex_ >= static_cast<int>(bodies_.size())) return;
    RigidBody& barrow = bodies_[static_cast<std::size_t>(wheelbarrowIndex_)];
    const BodySpec& barrowSpec = specs_[barrow.spec];
    if (barrow.asleep) {
        const bool movingCargo = std::any_of(
            bodies_.begin(), bodies_.end(), [&](const RigidBody& candidate) {
                return &candidate != &barrow && candidate.cachedBoundingRadius <= 0.26
                    && !candidate.asleep
                    && std::abs(candidate.position.x - barrow.position.x) <= 0.8
                    && std::abs(candidate.position.z - barrow.position.z) <= 1.1;
            });
        if (!movingCargo) return;
    }
    const Quat inverse = barrow.orientation.conjugate();
    const Real trayTop = barrowSpec.dimensions.y * 0.5;
    for (std::size_t index = 0; index < bodies_.size(); ++index) {
        if (static_cast<int>(index) == wheelbarrowIndex_) continue;
        RigidBody& body = bodies_[index];
        if (body.cachedBoundingRadius > 0.26 || body.stuckSurface || body.stuckTo >= 0)
            continue;
        if (std::abs(body.position.x - barrow.position.x) > 0.8
            || std::abs(body.position.z - barrow.position.z) > 1.1) continue;
        const Real cargoRadius = body.cachedBoundingRadius;
        Vec3 local = inverse.rotate(body.position - barrow.position);
        const Real xLimit = std::max(0.02, 0.275 - cargoRadius);
        const Real zMin = -0.255 + cargoRadius;
        const Real zMax = 0.485 - cargoRadius;
        if (std::abs(local.x) > 0.40 || local.z < -0.42 || local.z > 0.65
            || local.y - cargoRadius < trayTop - 0.08
            || local.y - cargoRadius > trayTop + 0.55) continue;
        const Vec3 original = local;
        local.x = clamp(local.x, -xLimit, xLimit);
        local.z = clamp(local.z, zMin, zMax);
        local.y = std::max(local.y, trayTop + cargoRadius);
        if ((local - original).lengthSquared() > 1.0e-12) {
            body.position = barrow.position + barrow.orientation.rotate(local);
        }
        const Vec3 relativeWorld = body.velocity - velocityAt(barrow, body.position);
        const Vec3 relativeLocal = inverse.rotate(relativeWorld);
        Vec3 deltaLocal{};
        if (original.x < -xLimit && relativeLocal.x < 0.0) deltaLocal.x = -relativeLocal.x;
        else if (original.x > xLimit && relativeLocal.x > 0.0) deltaLocal.x = -relativeLocal.x;
        if (original.z < zMin && relativeLocal.z < 0.0) deltaLocal.z = -relativeLocal.z;
        else if (original.z > zMax && relativeLocal.z > 0.0) deltaLocal.z = -relativeLocal.z;
        if (original.y < trayTop + cargoRadius && relativeLocal.y < 0.0)
            deltaLocal.y = -relativeLocal.y;
        const Vec3 wallDelta = barrow.orientation.rotate(deltaLocal);
        if (wallDelta.lengthSquared() > 1.0e-12) {
            body.velocity += wallDelta;
            barrow.velocity -= wallDelta * (dynamicMass(body)
                                             / std::max(dynamicMass(barrow), 1.0e-12));
        }
        Vec3 desired = barrow.velocity.horizontal() - body.velocity.horizontal();
        const Real maximumDelta = 4.0 * dt;
        const Real magnitude = desired.length();
        if (magnitude > maximumDelta) desired *= maximumDelta / magnitude;
        body.velocity += desired;
        barrow.velocity -= desired * (dynamicMass(body)
                                       / std::max(dynamicMass(barrow), 1.0e-12));
        body.grounded = true;
        wake(body);
    }
}

void PhysicsWorld::groundResistanceAndSleep(RigidBody& body, Real dt) {
    body.impactCooldown = std::max(0.0, body.impactCooldown - dt);
    if ((body.asleep && !body.held) || body.stuckSurface || body.stuckTo >= 0) return;
    const BodySpec& spec = specs_[body.spec];
    if (spec.key == "wheelbarrow") {
        const Vec3 forward = body.orientation.rotate({0.0, 0.0, 1.0})
            .horizontal().normalized({0.0, 0.0, 1.0});
        body.wheelAngle = std::fmod(
            body.wheelAngle + body.velocity.horizontal().dot(forward) * dt / 0.2032,
            2.0 * std::numbers::pi_v<Real>);
        if (body.wheelAngle < 0.0) body.wheelAngle += 2.0 * std::numbers::pi_v<Real>;
    }
    if (body.grounded && !body.held) {
        const Real scale = kDefaultRoomFriction > 0.0
            ? roomFriction_ / kDefaultRoomFriction : 1.0;
        const Vec3 horizontal = body.velocity.horizontal();
        const Real speed = horizontal.length();
        if (spec.key == "wheelbarrow") {
            const Vec3 forward = body.orientation.rotate({0.0, 0.0, 1.0})
                .horizontal().normalized({0.0, 0.0, 1.0});
            const Vec3 side{forward.z, 0.0, -forward.x};
            Real forwardSpeed = horizontal.dot(forward);
            Real sideSpeed = horizontal.dot(side);
            const Real forwardLoss = spec.rollingResistance * scale * gravity_ * dt;
            const Real sideLoss = 0.90 * scale * gravity_ * dt;
            forwardSpeed = std::copysign(std::max(0.0, std::abs(forwardSpeed) - forwardLoss),
                                         forwardSpeed);
            sideSpeed = std::copysign(std::max(0.0, std::abs(sideSpeed) - sideLoss),
                                      sideSpeed);
            const Vec3 result = forward * forwardSpeed + side * sideSpeed;
            body.velocity.x = result.x;
            body.velocity.z = result.z;
        } else if (speed > 1.0e-8) {
            const Real loss = spec.rollingResistance * scale * gravity_ * dt;
            const Real next = std::max(0.0, speed - loss);
            body.velocity.x *= next / speed;
            body.velocity.z *= next / speed;
        }
        const Real decay = std::max(0.0,
            1.0 - spec.rollingResistance * 2.0 * scale * dt);
        body.angularVelocity *= decay;
    }
    const bool pristineResting = body.pristine && !body.held;
    const Real linearSleepThreshold = pristineResting ? 0.12 : 0.025;
    const Real angularSleepThreshold = pristineResting ? 0.25 : 0.05;
    const Real sleepDelay = pristineResting ? 0.18 : 0.55;
    const bool quiet = body.velocity.length() < linearSleepThreshold
                    && body.angularVelocity.length() < angularSleepThreshold
                    && body.grounded;
    if (quiet && !body.held) {
        body.sleepTime += dt;
        if (body.sleepTime >= sleepDelay) {
            body.velocity = {};
            body.angularVelocity = {};
            body.asleep = true;
        }
    } else {
        body.sleepTime = 0.0;
        body.asleep = false;
    }
}

void PhysicsWorld::drainPendingWakes() {
    std::stable_sort(pendingWakes_.begin(), pendingWakes_.end(),
        [](const PendingWake& a, const PendingWake& b) {
            return a.impulse.lengthSquared() > b.impulse.lengthSquared();
        });
    int activeBricks = 0;
    int activeBulkTotal = 0;
    std::array<int, kBulkCategories.size()> activeBulk{};
    for (const auto& body : bodies_) {
        if (specs_[body.spec].key == "clay_brick" && !body.asleep) ++activeBricks;
        const int family = bulkFamilyIndex(specs_[body.spec]);
        if (family >= 0 && !body.asleep) {
            ++activeBulkTotal;
            ++activeBulk[static_cast<std::size_t>(family)];
        }
    }
    int newBricks = 0;
    int newBulk = 0;
    std::vector<PendingWake> deferred;
    for (const PendingWake& queued : pendingWakes_) {
        if (queued.body >= bodies_.size()) continue;
        RigidBody& body = bodies_[queued.body];
        const BodySpec& spec = specs_[body.spec];
        const bool brick = spec.key == "clay_brick";
        const int family = bulkFamilyIndex(spec);
        if (body.asleep && brick
            && (activeBricks >= kMaximumActiveBricks
                || newBricks >= kMaximumNewBrickWakes)) {
            deferred.push_back(queued);
            continue;
        }
        if (body.asleep && family >= 0) {
            const std::size_t slot = static_cast<std::size_t>(family);
            if (activeBulkTotal >= kMaximumActiveBulkBodies
                || activeBulk[slot] >= kBulkFamilyCaps[slot]
                || newBulk >= kMaximumNewBulkWakes) {
                deferred.push_back(queued);
                continue;
            }
        }
        if (body.asleep && brick) {
            ++activeBricks;
            ++newBricks;
        }
        if (body.asleep && family >= 0) {
            ++activeBulkTotal;
            ++activeBulk[static_cast<std::size_t>(family)];
            ++newBulk;
        }
        wake(body);
        applyImpulseRaw(spec, body, inverseMass(body),
                        queued.impulse, queued.point);
        body.lastImpulse = std::max(body.lastImpulse, queued.impulse.length());
    }
    pendingWakes_.swap(deferred);
}

void PhysicsWorld::beginFrameEvents() noexcept {
    impacts_.clear();
    collectingFrameEvents_ = true;
}

void PhysicsWorld::endFrameEvents() noexcept {
    collectingFrameEvents_ = false;
}

void PhysicsWorld::step(Real dt) {
    dt = clamp(dt, 0.0, kMaxFrameDt);
    if (dt <= 0.0) return;
    if (!collectingFrameEvents_) impacts_.clear();
    contacts_.clear();
    updateAttachments();
    integratePlayer(dt);
    if (heldBody_ >= 0 && heldBody_ < static_cast<int>(bodies_.size())) {
        holdConstraint(bodies_[static_cast<std::size_t>(heldBody_)]);
    }

    // Detach adhesive bodies before parallel integration so messages and host
    // payload mass are never mutated concurrently with an integration job.
    for (auto& body : bodies_) {
        if (!body.stuckSurface) continue;
        const BodySpec& spec = specs_[body.spec];
        const Real separating = std::max(0.0, body.force.dot(body.stuckNormal));
        if (separating <= spec.adhesionStrength) continue;
        body.stuckSurface = false;
        wake(body);
        addMessage((body.instanceLabel.empty() ? spec.name : body.instanceLabel)
                   + " peeled free from the room surface.");
    }
    // Detach adhesive children before parallel integration so host payload
    // mass is never mutated concurrently with a host force update.
    for (auto& body : bodies_) {
        if (body.stuckTo < 0 || body.stuckTo >= static_cast<int>(bodies_.size())) continue;
        const BodySpec& spec = specs_[body.spec];
        const Real separating = body.force.length();
        if (separating <= spec.adhesionStrength) continue;
        RigidBody& host = bodies_[static_cast<std::size_t>(body.stuckTo)];
        const Real transferred = spec.mass + spec.addedMass + body.attachedPayloadMass;
        host.attachedPayloadMass = std::max(0.0,
            host.attachedPayloadMass - transferred);
        body.stuckTo = -1;
        body.massCarriedByHost = false;
        wake(body);
        addMessage((body.instanceLabel.empty() ? spec.name : body.instanceLabel)
                   + " pulled free from "
                   + (host.instanceLabel.empty() ? specs_[host.spec].name
                                                  : host.instanceLabel)
                   + '.');
    }

    activeIndices_.clear();
    for (std::size_t index = 0; index < bodies_.size(); ++index) {
        if (!bodies_[index].asleep || bodies_[index].held) {
            activeIndices_.push_back(static_cast<std::uint16_t>(index));
        }
    }
    const auto integrateRange = [&](std::size_t begin, std::size_t end) {
        for (std::size_t slot = begin; slot < end; ++slot) {
            integrateBody(activeIndices_[slot], dt);
        }
    };
    const bool useParallelIntegration = cpuBackend_ == "maximum"
        || (cpuBackend_ == "auto"
            && activeIndices_.size() >= kParallelIntegrationBodies);
    if (useParallelIntegration) {
        executor_.parallelFor(activeIndices_.size(),
                              cpuBackend_ == "maximum" ? 1U : 4U,
                              integrateRange);
    } else {
        integrateRange(0, activeIndices_.size());
    }
    // Room impacts append ordered sound events, so this pass remains serial.
    for (const std::uint16_t index : activeIndices_) {
        advanceBodySwept(bodies_[index], dt);
    }

    buildBroadphasePairs(activeIndices_);
    lastBroadphaseCandidates_ = broadphasePairs_.size();
    int activeBricks = 0;
    for (const auto& body : bodies_) {
        if (specs_[body.spec].key == "clay_brick" && !body.asleep) ++activeBricks;
    }
    lastSolverIterations_ = (broadphasePairs_.size() > 60 || activeBricks > 8) ? 3
        : (broadphasePairs_.size() > 30 || activeBricks > 2) ? 4
        : kSolverIterations;

    std::vector<std::uint8_t> participating(bodies_.size(), 0);
    std::size_t activeContacts = 0;
    for (const Pair pair : broadphasePairs_) {
        RigidBody& first = bodies_[pair.first];
        RigidBody& second = bodies_[pair.second];
        if (first.asleep && second.asleep) continue;
        auto contact = findContact(pair.first, pair.second);
        if (!contact) continue;
        if (contact->toi < 1.0) {
            first.position = first.previousPosition
                + (first.position - first.previousPosition) * contact->toi;
            second.position = second.previousPosition
                + (second.position - second.previousPosition) * contact->toi;
        }
        (void)applyBodyImpulse(*contact, true);
        if (contact->toi < 1.0) {
            const Real remaining = dt * (1.0 - contact->toi);
            advanceBodySwept(first, remaining);
            advanceBodySwept(second, remaining);
        }
        contacts_.push_back(*contact);
        participating[pair.first] = 1;
        participating[pair.second] = 1;
        ++activeContacts;
    }
    lastActiveContacts_ = activeContacts;

    std::vector<std::uint16_t> playerCandidates;
    playerCandidates.reserve(32);
    for (std::size_t index = 0; index < bodies_.size(); ++index) {
        const RigidBody& body = bodies_[index];
        const Real reach = body.cachedBoundingRadius + kPlayerRadius + 0.35;
        const Real verticalLow = player_.position.y - body.cachedBoundingRadius;
        const Real verticalHigh = player_.position.y + kPlayerHeight
                                + body.cachedBoundingRadius;
        if (std::max(body.position.y, body.previousPosition.y) < verticalLow
            || std::min(body.position.y, body.previousPosition.y) > verticalHigh) {
            continue;
        }
        Real x = body.previousPosition.x - player_.position.x;
        Real z = body.previousPosition.z - player_.position.z;
        const Real motionX = body.position.x - body.previousPosition.x;
        const Real motionZ = body.position.z - body.previousPosition.z;
        const Real motion2 = motionX * motionX + motionZ * motionZ;
        if (motion2 > 1.0e-14) {
            const Real fraction = clamp(-(x * motionX + z * motionZ) / motion2,
                                        0.0, 1.0);
            x += motionX * fraction;
            z += motionZ * fraction;
        }
        if (x * x + z * z <= reach * reach) {
            playerCandidates.push_back(static_cast<std::uint16_t>(index));
            participating[index] = 1;
        }
    }
    for (const auto index : playerCandidates) resolvePlayerBody(index, dt);

    for (std::size_t index = 0; index < bodies_.size(); ++index) {
        if (participating[index] && !bodies_[index].asleep
            && !bodies_[index].stuckSurface && bodies_[index].stuckTo < 0) {
            projectBodyInsideRoom(bodies_[index]);
        }
    }
    projectPlayerInsideRoom();

    for (int iteration = 1; iteration < lastSolverIterations_; ++iteration) {
        for (const Contact& contact : contacts_) {
            (void)applyBodyImpulse(contact, false);
        }
        for (const auto index : playerCandidates) resolvePlayerBody(index, -1.0);
        for (std::size_t index = 0; index < bodies_.size(); ++index) {
            if (participating[index] && !bodies_[index].asleep
                && !bodies_[index].stuckSurface && bodies_[index].stuckTo < 0) {
                projectBodyInsideRoom(bodies_[index]);
            }
        }
        projectPlayerInsideRoom();
    }

    updateWheelbarrowCargo(dt);
    updateAttachments();
    for (const auto index : activeIndices_) participating[index] = 1;
    for (std::size_t index = 0; index < bodies_.size(); ++index) {
        if (!bodies_[index].asleep) participating[index] = 1;
        if (!participating[index]) continue;
        groundResistanceAndSleep(bodies_[index], dt);
        const RigidBody& body = bodies_[index];
        if (!body.asleep && (!body.position.finite() || !body.velocity.finite()
                            || !body.angularVelocity.finite())) {
            throw std::runtime_error("non-finite authoritative rigid-body state");
        }
    }
    drainPendingWakes();
    // Keep the kinematic player invariant exact even if future physics features
    // add another reaction path. Yaw and pitch deliberately remain untouched.
    enforcePlayerPositionLock();
    updateGalileoExperiment(dt);
    updateGravityInterceptExperiment(dt);
    simulationTime_ += dt;
}

RuntimeOptions parseArguments(int argc, char** argv) {
    RuntimeOptions options;
    const auto requireValue = [&](int& index, std::string_view name) -> std::string {
        if (index + 1 >= argc) {
            throw std::invalid_argument(std::string(name) + " requires a value");
        }
        return argv[++index];
    };
    const auto parseInteger = [](std::string_view value, std::string_view name,
                                 int minimum, int maximum) {
        int parsed = 0;
        const char* const begin = value.data();
        const char* const end = begin + value.size();
        const auto [position, error] = std::from_chars(begin, end, parsed);
        if (error != std::errc{} || position != end
            || parsed < minimum || parsed > maximum) {
            throw std::invalid_argument(std::string(name)
                + " must be an integer in " + std::to_string(minimum)
                + ".." + std::to_string(maximum));
        }
        return parsed;
    };
    const auto parseSeed = [](std::string_view value) {
        std::uint64_t parsed = 0;
        const char* const begin = value.data();
        const char* const end = begin + value.size();
        const auto [position, error] = std::from_chars(begin, end, parsed);
        if (error != std::errc{} || position != end
            || parsed > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "--seed must be an integer in 0..4294967295");
        }
        return static_cast<std::uint32_t>(parsed);
    };
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--check") options.check = true;
        else if (argument == "--vk-check" || argument == "--gl-check") {
            options.glCheck = true;
        }
        else if (argument == "--no-audio") options.noAudio = true;
        else if (argument == "--maximum-throughput") options.maximumThroughput = true;
        else if (argument == "--stress-pallet") options.stressPallet = true;
        else if (argument == "--seed") {
            options.seed = parseSeed(requireValue(index, argument));
        } else if (argument == "--frames") {
            options.frames = parseInteger(requireValue(index, argument), argument,
                                           0, std::numeric_limits<int>::max());
        } else if (argument == "--view") {
            options.view = requireValue(index, argument);
            if (options.view != "start" && options.view != "pallet"
                && options.view != "exhibits" && options.view != "warehouse") {
                throw std::invalid_argument(
                    "--view must be start, pallet, exhibits, or warehouse");
            }
        } else if (argument == "--capture") {
            options.capturePath = requireValue(index, argument);
            if (options.capturePath.empty()) {
                throw std::invalid_argument("--capture path cannot be empty");
            }
        } else if (argument == "--quality") {
            options.quality = requireValue(index, argument);
            if (options.quality != "auto" && options.quality != "safe"
                && options.quality != "balanced" && options.quality != "ultra") {
                throw std::invalid_argument(
                    "--quality must be auto, safe, balanced, or ultra");
            }
        } else if (argument == "--cpu-cores") {
            options.cpuCores = parseInteger(requireValue(index, argument),
                                             argument, 1, 4);
        } else if (argument == "--cpu-backend") {
            options.cpuBackend = requireValue(index, argument);
            if (options.cpuBackend != "scalar" && options.cpuBackend != "auto"
                && options.cpuBackend != "maximum") {
                throw std::invalid_argument(
                    "--cpu-backend must be scalar, auto, or maximum");
            }
        } else if (argument == "--gpu-physics") {
            options.gpuPhysics = requireValue(index, argument);
            if (options.gpuPhysics != "off" && options.gpuPhysics != "auto"
                && options.gpuPhysics != "maximum") {
                throw std::invalid_argument(
                    "--gpu-physics must be off, auto, or maximum");
            }
        } else if (argument == "--flashlight") {
            options.flashlight = requireValue(index, argument);
            if (options.flashlight != "off" && options.flashlight != "on") {
                throw std::invalid_argument("--flashlight must be off or on");
            }
        } else if (argument == "--shadows") {
            options.shadows = requireValue(index, argument);
            if (options.shadows != "off" && options.shadows != "on") {
                throw std::invalid_argument("--shadows must be off or on");
            }
        } else if (argument == "--overhead-lights") {
            options.overheadLights = requireValue(index, argument);
            if (options.overheadLights != "off"
                && options.overheadLights != "on") {
                throw std::invalid_argument(
                    "--overhead-lights must be off or on");
            }
        } else if (argument == "--textures") {
            options.textures = requireValue(index, argument);
            if (options.textures != "off" && options.textures != "on") {
                throw std::invalid_argument("--textures must be off or on");
            }
        } else if (argument == "--bump-mapping") {
            options.bumpMapping = requireValue(index, argument);
            if (options.bumpMapping != "off"
                && options.bumpMapping != "on") {
                throw std::invalid_argument(
                    "--bump-mapping must be off or on");
            }
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << kTitle << "\n"
                << "  --check --vk-check --capture PATH --frames N --seed N\n"
                << "  --view start|pallet|exhibits|warehouse --quality auto|safe|balanced|ultra\n"
                << "  --cpu-cores 1..4 --cpu-backend scalar|auto|maximum\n"
                << "  --gpu-physics off|auto|maximum --flashlight on|off --shadows on|off\n"
                << "  --overhead-lights on|off --no-audio\n"
                << "  --textures on|off --bump-mapping on|off\n"
                << "  --maximum-throughput --stress-pallet\n"
                << "  Controls: C toggles the head-mounted flashlight; V toggles the\n"
                << "  overhead grid. F5 restores both configured startup states.\n"
                << "  F1 help; F2 HUD; F3 pause; F4 single-step; F6 Galileo;\n"
                << "  F7 trajectory echoes; F8 Gravity-Proof Trick Shot;\n"
                << "  F9 kinetic lens; F10 resonance waves; F11 material view;\n"
                << "  F12 adaptive normal detail; Q Echo Pulse.\n"
                << "  --shadows controls\n"
                << "  flashlight shadows (on by default; inactive while the light is off).\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    return options;
}

bool runSelfCheck(bool verbose) {
    std::vector<std::string> passed;
    std::vector<std::string> failed;
    const auto verify = [&](std::string label, bool condition,
                            std::string detail = {}) {
        if (condition) passed.push_back(std::move(label));
        else failed.push_back(std::move(label) + (detail.empty() ? "" : ": " + detail));
    };
    const auto near = [](Real a, Real b, Real tolerance = 1.0e-9) {
        return std::abs(a - b) <= tolerance;
    };
    const auto sameVec = [](const Vec3& a, const Vec3& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    };
    const auto sameQuat = [](const Quat& a, const Quat& b) {
        return a.w == b.w && a.x == b.x && a.y == b.y && a.z == b.z;
    };
    // Most probes exercise one isolated mechanic. Quiescing their unrelated
    // warehouse inventory keeps --check deterministic and fast while the
    // dedicated initialization assertions below still enforce an all-awake
    // production reset.
    const auto sleepAllBodies = [](PhysicsWorld& probe) {
        for (RigidBody& body : probe.bodies()) {
            body.velocity = {};
            body.angularVelocity = {};
            body.force = {};
            body.torque = {};
            body.sleepTime = 0.0;
            body.asleep = true;
            body.held = false;
        }
    };

    // Configuration and numeric architecture (1-15).
    verify("room dimensions", kRoomWidth == 100.0 && kRoomLength == 100.0
                               && kRoomHeight == 10.0);                         // 1
    verify("two-metre player", kPlayerHeight == 2.0
                                && kPlayerEyeHeight < kPlayerHeight);            // 2
    verify("Earth gravity default", near(kEarthGravity, 9.80665, 1.0e-12));     // 3
    verify("gravity range", kGravityMin < kEarthGravity
                             && kEarthGravity < kGravityMax);                    // 4
    verify("friction range", kFrictionMin <= kDefaultRoomFriction
                              && kDefaultRoomFriction <= kFrictionMax);          // 5
    verify("throw range", kThrowForceMin == 1.0
                           && kThrowForceMax == 1'000'000.0);                    // 6
    verify("eight-pass solver", kSolverIterations == 8);                        // 7
    verify("120 Hz fixed step", kPhysicsHz == 120
                                && near(kFixedDt, 1.0 / 120.0, 1.0e-15));        // 8
    verify("60/30 frame contract", kTargetFps == 60 && kMinimumFps == 30);       // 9
    verify("exact watermark", kWatermark
        == "Made by OpenAI ChatGPT Codex 5.6 Sol Ultra");                       // 10
    verify("native title", kTitle.find("C++20 Vulkan 1.3") != std::string_view::npos); // 11
    verify("double authoritative scalar", sizeof(Real) == sizeof(double));      // 12
    verify("throw work stroke", near(kThrowStroke, 0.75));                      // 13
    verify("six-metre pickup", near(kPickupReach, 6.0));                        // 14
    verify("physical air density", near(kAirDensity, 1.229));                   // 15

    PhysicsWorld world(2468, 3);
    const auto& specs = world.specs();
    const auto findSpec = [&](std::string_view key) -> const BodySpec& {
        const auto iterator = std::find_if(specs.begin(), specs.end(),
            [&](const BodySpec& spec) { return spec.key == key; });
        if (iterator == specs.end()) throw std::logic_error("self-check missing spec");
        return *iterator;
    };
    // Catalog fidelity (16-38).
    verify("thirty-one textured material variants", specs.size() == 31
        && std::all_of(specs.begin(), specs.end(), [](const BodySpec& spec) {
            return spec.surfaceMaterial != SurfaceMaterial::Neutral;
        })
        && findSpec("concrete").surfaceMaterial == SurfaceMaterial::Concrete
        && findSpec("steel").surfaceMaterial == SurfaceMaterial::Metal
        && findSpec("rubber_brick").surfaceMaterial == SurfaceMaterial::Rubber
        && findSpec("timber_log").surfaceMaterial == SurfaceMaterial::Wood
        && findSpec("plush_bear").surfaceMaterial == SurfaceMaterial::Plush);    // 16
    verify("seven calibration specs", std::count_if(specs.begin(), specs.end(),
        [](const BodySpec& spec) { return spec.category == "calibration"; }) == 7); // 17
    verify("medicine masses", findSpec("medicine_1").mass == 1.0
        && findSpec("medicine_3").mass == 3.0
        && findSpec("medicine_10").mass == 10.0);                               // 18
    verify("dodgeball midpoint mass", near(findSpec("dodge").mass, 0.310));    // 19
    verify("dodgeball midpoint diameter",
           near(findSpec("dodge").diameter, 0.660 / std::numbers::pi_v<Real>)); // 20
    verify("steel density-derived mass", near(findSpec("steel").mass,
                                                58.191484683, 3.0e-6));          // 21
    verify("concrete density-derived mass", near(findSpec("concrete").mass,
                                                   17.791027164, 3.0e-6));       // 22
    verify("rubber brick density-derived mass",
           near(findSpec("rubber_brick").mass, 0.93594912, 1.0e-10));          // 23
    verify("ash bat profile", near(findSpec("wood_bat").mass, 0.8788)
        && near(findSpec("wood_bat").dimensions.x, 0.8636));                   // 24
    verify("wheelbarrow profile", near(findSpec("wheelbarrow").mass, 17.28)
        && near(findSpec("wheelbarrow").dimensions.z, 1.492));                 // 25
    verify("helium profile", near(findSpec("helium_balloon").buoyancyVolume,
                                   0.4248));                                    // 26
    verify("noodle profile", near(findSpec("foam_noodle").dimensions.x,
                                   1.1938));                                    // 27
    verify("goo finite adhesion", findSpec("sticky_goo").adhesionStrength == 25.0); // 28
    verify("ceramic profile", near(findSpec("ceramic_marble").diameter, 0.050)); // 29
    verify("clay brick profile", findSpec("clay_brick").dimensions.x == 0.194
        && findSpec("clay_brick").mass == 1.90);                               // 30
    verify("EPAL pallet profile", findSpec("wood_pallet").dimensions.x == 1.2
        && findSpec("wood_pallet").mass == 25.0);                              // 31
    verify("ten plush variants", std::count_if(specs.begin(), specs.end(),
        [](const BodySpec& spec) { return spec.category == "plush"; }) == 10);   // 32
    verify("positive inertia", std::all_of(specs.begin(), specs.end(),
        [](const BodySpec& spec) {
            const Vec3 inertia = spec.inertiaDiagonal();
            return inertia.x > 0.0 && inertia.y > 0.0 && inertia.z > 0.0;
        }));                                                                    // 33
    verify("inflated shell inertia", findSpec("dodge").thinShell
        && findSpec("helium_balloon").thinShell);                              // 34
    verify("elasticity ordering", findSpec("dodge").restitution
        > findSpec("steel").restitution
        && findSpec("steel").restitution > findSpec("medicine_1").restitution
        && findSpec("medicine_1").restitution > findSpec("concrete").restitution); // 35
    const BodySpec& bowling = findSpec("bowling_ball");
    const Real expectedBowlingVolume = 4.0 * std::numbers::pi_v<Real>
        * std::pow(0.2183 * 0.5, 3.0) / 3.0;
    verify("realistic fifteen-pound bowling profile",
           bowling.shape == Shape::Sphere
        && bowling.renderKind == RenderKind::Bowling
        && near(bowling.mass, 6.80388555, 1.0e-9)
        && near(bowling.diameter, 0.2183, 1.0e-12)
        && near(bowling.density, bowling.mass / expectedBowlingVolume, 1.0e-9)
        && near(bowling.restitution, 0.18)
        && near(bowling.friction, 0.24));                                       // 36
    const BodySpec& timber = findSpec("timber_log");
    const Real expectedTimberMass = std::numbers::pi_v<Real>
        * std::pow(0.30 * 0.5, 2.0) * 2.40 * 530.0;
    verify("realistic Douglas-fir timber profile",
           timber.shape == Shape::Box
        && timber.renderKind == RenderKind::TimberLog
        && near(timber.dimensions.x, 2.40)
        && near(timber.dimensions.y, 0.30)
        && near(timber.dimensions.z, 0.30)
        && near(timber.mass, expectedTimberMass, 1.0e-9)
        && near(timber.density, 530.0));                                        // 37
    verify("bulk variants preserve source physics",
           near(findSpec("pallet_dodge").mass, findSpec("dodge").mass)
        && near(findSpec("pallet_dodge").diameter, findSpec("dodge").diameter)
        && near(findSpec("pallet_goo").mass, findSpec("sticky_goo").mass)
        && near(findSpec("pallet_goo").adhesionStrength,
                findSpec("sticky_goo").adhesionStrength)
        && near(findSpec("pallet_marble").mass,
                findSpec("ceramic_marble").mass)
        && findSpec("pallet_marble").renderKind == RenderKind::Ceramic
        && near(findSpec("bulk_pallet").mass, findSpec("wood_pallet").mass)
        && findSpec("bulk_pallet").renderKind == RenderKind::Pallet);           // 38

    const auto countCategory = [&](std::string_view category) {
        return std::count_if(world.bodies().begin(), world.bodies().end(),
            [&](const RigidBody& body) { return specs[body.spec].category == category; });
    };
    const auto countKey = [&](std::string_view key) {
        return std::count_if(world.bodies().begin(), world.bodies().end(),
            [&](const RigidBody& body) { return specs[body.spec].key == key; });
    };
    const auto highestTop = [&](std::string_view category) {
        Real top = -std::numeric_limits<Real>::infinity();
        for (const RigidBody& body : world.bodies()) {
            if (specs[body.spec].category != category) continue;
            top = std::max(top, body.position.y
                + world.supportExtent(body, Vec3{0.0, 1.0, 0.0}));
        }
        return top;
    };
    // Population and reset geometry (39-66).
    verify("exact 4,487-body population", world.bodies().size() == kExpectedBodyCount); // 39
    verify("player floor start", world.player().position.x == 10.0
        && world.player().position.y == 0.0 && world.player().position.z == 10.0
        && world.player().grounded);                                             // 40
    verify("original lineup positions", [&] {
        for (std::size_t index = 0; index < 7; ++index) {
            if (!near(world.bodies()[index].position.x,
                      7.0 + static_cast<Real>(index))
                || !near(world.bodies()[index].position.z, 15.0)) return false;
        }
        return true;
    }());                                                                        // 41
    verify("original lineup non-overlap", [&] {
        for (std::size_t first = 0; first < 7; ++first)
            for (std::size_t second = first + 1; second < 7; ++second)
                if ((world.bodies()[second].position - world.bodies()[first].position).length()
                    <= world.bodies()[first].cachedBoundingRadius
                       + world.bodies()[second].cachedBoundingRadius) return false;
        return true;
    }());                                                                        // 42
    verify("seven calibration bodies", countCategory("calibration") == 7);     // 43
    verify("one bat body", countCategory("bat") == 1);                         // 44
    verify("one wheelbarrow body", countCategory("wheelbarrow") == 1);         // 45
    verify("one balloon body", countCategory("balloon") == 1);                 // 46
    verify("five noodles", countCategory("noodles") == 5);                    // 47
    verify("ten goo blobs", countCategory("goo") == 10);                      // 48
    verify("twenty marbles", countCategory("marbles") == 20);                 // 49
    verify("five hundred bricks", countCategory("bricks") == 500);            // 50
    verify("one pallet", countCategory("pallet") == 1);                       // 51
    verify("five stuffed animals", countCategory("plush") == 5);              // 52
    std::set<std::uint16_t> plushKinds;
    for (const auto& body : world.bodies())
        if (specs[body.spec].category == "plush") plushKinds.insert(body.spec);
    verify("unique stuffed animals", plushKinds.size() == 5);                   // 53
    verify("975 kg palletized load", near(
        findSpec("clay_brick").mass * kClayBrickCount
            + findSpec("wood_pallet").mass, 975.0));                            // 54
    verify("three hundred bowling balls",
           countKey("bowling_ball") == kPalletBowlingBallCount
        && countCategory("bulk_bowling") == kPalletBowlingBallCount);           // 55
    verify("three hundred pallet dodgeballs",
           countKey("pallet_dodge") == kPalletDodgeballCount
        && countCategory("bulk_dodge") == kPalletDodgeballCount);               // 56
    verify("three hundred pallet goo globs",
           countKey("pallet_goo") == kPalletGooCount
        && countCategory("bulk_goo") == kPalletGooCount);                       // 57
    verify("three thousand pallet marbles",
           countKey("pallet_marble") == kPalletMarbleCount
        && countCategory("bulk_marble") == kPalletMarbleCount);                 // 58
    verify("thirty warehouse pallet bodies",
           countKey("bulk_pallet")
               == kWarehousePalletBaseCount + kNestedPalletCount
        && countCategory("bulk_pallet")
               == kWarehousePalletBaseCount + kNestedPalletCount);              // 59
    const auto basePalletCount = std::count_if(
        world.bodies().begin(), world.bodies().end(), [&](const RigidBody& body) {
            return specs[body.spec].key == "bulk_pallet"
                && body.instanceLabel.ends_with("Pallet Base");
        });
    const auto nestedPalletCount = std::count_if(
        world.bodies().begin(), world.bodies().end(), [&](const RigidBody& body) {
            return specs[body.spec].key == "bulk_pallet"
                && body.instanceLabel.starts_with("Stacked EPAL Pallet ");
        });
    verify("five bases plus twenty-five pallet payload",
           basePalletCount == kWarehousePalletBaseCount
        && nestedPalletCount == kNestedPalletCount);                             // 60
    verify("six timber logs", countKey("timber_log") == kTimberLogCount
        && countCategory("timber") == kTimberLogCount);                         // 61
    const auto warehouseBodyCount = std::count_if(
        world.bodies().begin(), world.bodies().end(), [&](const RigidBody& body) {
            return isBulkStock(specs[body.spec])
                || specs[body.spec].category == "timber";
        });
    verify("exact warehouse addition population",
           warehouseBodyCount == kExpectedBodyCount - kOriginalBodyCount);      // 62
    verify("all bodies initialize awake pristine and zeroed", std::all_of(
        world.bodies().begin(), world.bodies().end(), [&](const RigidBody& body) {
            return !body.asleep && body.pristine && !body.held
                && !body.stuckSurface && body.stuckTo < 0
                && !body.massCarriedByHost
                && body.velocity.lengthSquared() == 0.0
                && body.angularVelocity.lengthSquared() == 0.0
                && body.force.lengthSquared() == 0.0
                && body.torque.lengthSquared() == 0.0
                && body.sleepTime == 0.0;
        }));                                                                     // 63
    verify("warehouse bodies fit room bounds", std::all_of(
        world.bodies().begin(), world.bodies().end(), [&](const RigidBody& body) {
            if (!isBulkStock(specs[body.spec])
                && specs[body.spec].category != "timber") return true;
            const Real extentX = world.supportExtent(body, {1.0, 0.0, 0.0});
            const Real extentY = world.supportExtent(body, {0.0, 1.0, 0.0});
            const Real extentZ = world.supportExtent(body, {0.0, 0.0, 1.0});
            constexpr Real tolerance = 1.0e-9;
            return body.position.x - extentX >= -tolerance
                && body.position.x + extentX <= kRoomWidth + tolerance
                && body.position.y - extentY >= -tolerance
                && body.position.y + extentY <= kRoomHeight + tolerance
                && body.position.z - extentZ >= -tolerance
                && body.position.z + extentZ <= kRoomLength + tolerance;
        }));                                                                     // 64
    const auto expectedSphereStackTop = [&](std::string_view key, int count,
                                             int perLayer, Real gap) {
        const BodySpec& spec = findSpec(key);
        const int layers = (count + perLayer - 1) / perLayer;
        return findSpec("bulk_pallet").dimensions.y + spec.diameter
            + static_cast<Real>(layers - 1) * (spec.diameter + gap);
    };
    verify("warehouse sphere stack heights",
           near(highestTop("bulk_bowling"),
                expectedSphereStackTop("bowling_ball",
                    kPalletBowlingBallCount, 15, 0.003), 1.0e-9)
        && near(highestTop("bulk_dodge"),
                expectedSphereStackTop("pallet_dodge",
                    kPalletDodgeballCount, 15, 0.003), 1.0e-9)
        && near(highestTop("bulk_goo"),
                expectedSphereStackTop("pallet_goo",
                    kPalletGooCount, 60, 0.003), 1.0e-9)
        && near(highestTop("bulk_marble"),
                expectedSphereStackTop("pallet_marble",
                    kPalletMarbleCount, 300, 0.002), 1.0e-9));                   // 65
    const Real expectedPalletStackTop = findSpec("bulk_pallet").dimensions.y
        + kNestedPalletCount * findSpec("bulk_pallet").dimensions.y
        + (kNestedPalletCount - 1) * 0.002;
    verify("nested-pallet and timber stack heights",
           near(highestTop("bulk_pallet"), expectedPalletStackTop, 1.0e-9)
        && near(highestTop("timber"), timber.dimensions.y * 2.0 + 0.005,
                1.0e-9)
        && highestTop("bulk_pallet") < kRoomHeight
        && highestTop("timber") < kRoomHeight);                                // 66
    verify("all warehouse labels are populated", std::all_of(
        world.bodies().begin(), world.bodies().end(), [&](const RigidBody& body) {
            return (!isBulkStock(specs[body.spec])
                    && specs[body.spec].category != "timber")
                || !body.instanceLabel.empty();
        }));                                                                     // 67

    // Controls, interactions, integration, wake budgets, and determinism (68-115).
    const Real gravityBefore = world.gravity();
    world.adjustGravity(1);
    verify("gravity increment", near(world.gravity(), gravityBefore + kGravityStep)); // 68
    verify("global tuning releases all-awake initial support", std::all_of(
        world.bodies().begin(), world.bodies().end(), [](const RigidBody& body) {
            return !body.asleep && !body.pristine;
        }));                                                                     // 69

    PhysicsWorld localBulkWakeProbe(2'469, 0);
    const auto countAwakeBulk = [](const PhysicsWorld& probe) {
        return std::count_if(probe.bodies().begin(), probe.bodies().end(),
            [&](const RigidBody& body) {
                return isBulkStock(probe.specs()[body.spec]) && !body.asleep;
            });
    };
    for (RigidBody& body : localBulkWakeProbe.bodies()) {
        if (!isBulkStock(localBulkWakeProbe.specs()[body.spec])) continue;
        body.asleep = true;
        body.sleepTime = 0.0;
    }
    localBulkWakeProbe.wakeNearby({54.0, 2.0, 44.0}, 5.0,
                                  kMaximumNewBulkWakes);
    const auto firstLocalWakeCount = countAwakeBulk(localBulkWakeProbe);
    localBulkWakeProbe.wakeNearby({54.0, 2.0, 44.0}, 5.0,
                                  kMaximumNewBulkWakes);
    const auto secondLocalWakeCount = countAwakeBulk(localBulkWakeProbe);
    verify("first local bulk wake stage is capped",
           firstLocalWakeCount == kMaximumNewBulkWakes);                        // 70
    verify("second local bulk wake stage is capped",
           secondLocalWakeCount - firstLocalWakeCount == kMaximumNewBulkWakes); // 71

    PhysicsWorld globalBulkWakeProbe(2'470, 0);
    for (RigidBody& body : globalBulkWakeProbe.bodies()) {
        if (!isBulkStock(globalBulkWakeProbe.specs()[body.spec])) continue;
        body.asleep = true;
        body.pristine = false;
    }
    globalBulkWakeProbe.wakeAll();
    std::array<int, kBulkCategories.size()> awakeBulkByFamily{};
    for (const RigidBody& body : globalBulkWakeProbe.bodies()) {
        if (body.asleep) continue;
        const int family = bulkFamilyIndex(globalBulkWakeProbe.specs()[body.spec]);
        if (family >= 0) ++awakeBulkByFamily[static_cast<std::size_t>(family)];
    }
    const int totalAwakeBulk = std::accumulate(
        awakeBulkByFamily.begin(), awakeBulkByFamily.end(), 0);
    verify("global bulk wake ceiling",
           totalAwakeBulk == kMaximumActiveBulkBodies);                         // 72
    constexpr std::array<int, 5> expectedWakeProfile{32, 32, 32, 27, 5};
    verify("deterministic per-family bulk wake ceilings",
           awakeBulkByFamily == expectedWakeProfile
        && std::equal(awakeBulkByFamily.begin(), awakeBulkByFamily.end(),
                      kBulkFamilyCaps.begin(),
                      [](int active, int cap) { return active <= cap; }));       // 73

    world.adjustGravity(-1);
    verify("gravity decrement", near(world.gravity(), gravityBefore));          // 74
    const Real frictionBefore = world.roomFriction();
    world.adjustFriction(1);
    verify("friction increment", near(world.roomFriction(), frictionBefore + kFrictionStep)); // 75
    world.adjustFriction(-1);
    verify("friction decrement", near(world.roomFriction(), frictionBefore));   // 76
    world.adjustThrowForce(1);
    verify("1-2-5 force increment", world.throwForce() == 2.0);                 // 77
    world.adjustThrowForce(-1);
    verify("1-2-5 force decrement", world.throwForce() == 1.0);                 // 78
    sleepAllBodies(world);
    world.setMoveInput(2.0, -2.0);
    verify("movement input clamp", world.player().moveForward == 1.0
        && world.player().moveStrafe == -1.0);                                   // 79
    world.setMoveInput(0.0, 0.0);
    verify("first jump accepted", world.jump());                                // 80
    verify("air jump rejected", !world.jump());                                // 81
    Real peak = world.player().position.y;
    for (int tick = 0; tick < 300 && !world.player().grounded; ++tick) {
        world.step(kFixedDt);
        peak = std::max(peak, world.player().position.y);
    }
    verify("one-metre jump apex", std::abs(peak - kJumpHeight) < 0.001,
           "peak=" + std::to_string(peak));                                     // 82
    verify("landing event retained", world.player().landingSpeed > 0.0);        // 83
    verify("simulation clock advances", world.simulationTime() > 0.0);          // 84
    PhysicsWorld scalarBackendProbe(31, 3, "scalar");
    sleepAllBodies(scalarBackendProbe);
    scalarBackendProbe.step(kFixedDt);
    verify("parallel helper budget", world.helperCount() <= 3
        && scalarBackendProbe.helperCount() == 3
        && scalarBackendProbe.cpuBackend() == "scalar");                       // 85
    verify("finite settled state", std::all_of(world.bodies().begin(),
        world.bodies().end(), [](const RigidBody& body) {
            return body.position.finite() && body.velocity.finite()
                && body.angularVelocity.finite();
        }));                                                                     // 86
    verify("room containment", std::all_of(world.bodies().begin(),
        world.bodies().end(), [](const RigidBody& body) {
            return body.position.x >= 0.0 && body.position.x <= kRoomWidth
                && body.position.y >= 0.0 && body.position.y <= kRoomHeight
                && body.position.z >= 0.0 && body.position.z <= kRoomLength;
        }));                                                                     // 87
    verify("bounded broadphase", world.broadphaseCandidates() < 1'000);         // 88
    PhysicsWorld interaction(4, 0);
    const int aimed = interaction.raycastBody(interaction.player().eye(),
                                               interaction.player().forward());
    verify("crosshair raycast", aimed >= 0);                                    // 89
    verify("right-click pickup semantic", interaction.pickupOrDrop());          // 90
    verify("held body index", interaction.heldBody() >= 0);                     // 91
    verify("right-click drop semantic", interaction.pickupOrDrop()
        && interaction.heldBody() < 0);                                          // 92
    verify("throw requires held body", !interaction.throwHeld());               // 93
    const bool pickedAgain = interaction.pickupOrDrop();
    interaction.adjustThrowForce(1);
    const Vec3 playerVelocityBefore = interaction.player().velocity;
    const int thrownIndex = interaction.heldBody();
    const bool threw = pickedAgain && interaction.throwHeld();
    verify("held body throw", threw && thrownIndex >= 0);                        // 94
    verify("throw launches body", threw
        && interaction.bodies()[static_cast<std::size_t>(thrownIndex)].velocity.length() > 0.0); // 95
    verify("throw applies recoil", threw
        && (interaction.player().velocity - playerVelocityBefore).length() > 0.0); // 96
    interaction.adjustGravity(1);
    interaction.adjustFriction(1);
    interaction.togglePlayerPositionLock();
    interaction.player().position = {19.0, 1.25, 23.0};
    interaction.reset();
    verify("F5 restores canonical settings and player",
           interaction.gravity() == kEarthGravity
        && interaction.roomFriction() == kDefaultRoomFriction
        && interaction.throwForce() == kThrowForceMin
        && !interaction.playerPositionLocked()
        && sameVec(interaction.player().position, Player{}.position)
        && sameVec(interaction.player().previousPosition, Player{}.previousPosition)
        && sameVec(interaction.player().velocity, {})
        && interaction.player().yaw == Player{}.yaw
        && interaction.player().pitch == Player{}.pitch
        && interaction.player().grounded);                                      // 97
    verify("reset restores population", interaction.bodies().size()
        == static_cast<std::size_t>(kExpectedBodyCount));                        // 98
    char executable[] = "nec";
    char framesFlag[] = "--frames";
    char framesValue[] = "12";
    char maximumFlag[] = "--maximum-throughput";
    char flashlightFlag[] = "--flashlight";
    char flashlightOn[] = "on";
    char shadowsFlag[] = "--shadows";
    char shadowsOff[] = "off";
    char overheadFlag[] = "--overhead-lights";
    char overheadOff[] = "off";
    char texturesFlag[] = "--textures";
    char texturesOff[] = "off";
    char bumpFlag[] = "--bump-mapping";
    char bumpOff[] = "off";
    char* arguments[]{executable, framesFlag, framesValue, maximumFlag,
                      flashlightFlag, flashlightOn, shadowsFlag, shadowsOff,
                      overheadFlag, overheadOff, texturesFlag, texturesOff,
                      bumpFlag, bumpOff};
    const RuntimeOptions parsed = parseArguments(14, arguments);
    char invalidFlashlight[] = "invalid";
    char* invalidArguments[]{executable, flashlightFlag, invalidFlashlight};
    bool rejectedInvalidFlashlight = false;
    try {
        (void)parseArguments(3, invalidArguments);
    } catch (const std::invalid_argument&) {
        rejectedInvalidFlashlight = true;
    }
    char invalidShadows[] = "invalid";
    char* invalidShadowArguments[]{executable, shadowsFlag, invalidShadows};
    bool rejectedInvalidShadows = false;
    try {
        (void)parseArguments(3, invalidShadowArguments);
    } catch (const std::invalid_argument&) {
        rejectedInvalidShadows = true;
    }
    char invalidOverhead[] = "invalid";
    char* invalidOverheadArguments[]{executable, overheadFlag, invalidOverhead};
    bool rejectedInvalidOverhead = false;
    try {
        (void)parseArguments(3, invalidOverheadArguments);
    } catch (const std::invalid_argument&) {
        rejectedInvalidOverhead = true;
    }
    char invalidTextures[] = "invalid";
    char* invalidTextureArguments[]{executable, texturesFlag, invalidTextures};
    bool rejectedInvalidTextures = false;
    try {
        (void)parseArguments(3, invalidTextureArguments);
    } catch (const std::invalid_argument&) {
        rejectedInvalidTextures = true;
    }
    char invalidBump[] = "invalid";
    char* invalidBumpArguments[]{executable, bumpFlag, invalidBump};
    bool rejectedInvalidBump = false;
    try {
        (void)parseArguments(3, invalidBumpArguments);
    } catch (const std::invalid_argument&) {
        rejectedInvalidBump = true;
    }
    verify("command-line profile", parsed.frames == 12
        && parsed.maximumThroughput && parsed.flashlight == "on"
        && RuntimeOptions{}.flashlight == "off"
        && parsed.shadows == "off" && RuntimeOptions{}.shadows == "on"
        && parsed.overheadLights == "off"
        && RuntimeOptions{}.overheadLights == "on"
        && parsed.textures == "off" && RuntimeOptions{}.textures == "on"
        && parsed.bumpMapping == "off"
        && RuntimeOptions{}.bumpMapping == "on"
        && rejectedInvalidFlashlight && rejectedInvalidShadows
        && rejectedInvalidOverhead && rejectedInvalidTextures
        && rejectedInvalidBump);                                                  // 99
    ParallelExecutor handoffProbe(3);
    std::array<std::uint32_t, 9> completionStamp{};
    bool completionHandoff = true;
    for (std::uint32_t generation = 1; generation <= 1'024; ++generation) {
        completionStamp.fill(0U);
        handoffProbe.parallelFor(completionStamp.size(), 4,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t index = begin; index < end; ++index) {
                    completionStamp[index] = generation;
                }
            });
        completionHandoff = completionHandoff
            && std::all_of(completionStamp.begin(), completionStamp.end(),
                [generation](std::uint32_t value) { return value == generation; });
    }
    verify("parallel completion handoff", completionHandoff);                    // 100

    const auto samePosition = [](const Vec3& a, const Vec3& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    };
    PhysicsWorld lockProbe(77, 0);
    sleepAllBodies(lockProbe);
    verify("player physics mode default", !lockProbe.playerPositionLocked());    // 101
    const Vec3 freeStart = lockProbe.player().position;
    lockProbe.setMoveInput(1.0, 0.0);
    for (int tick = 0; tick < 24; ++tick) lockProbe.step(kFixedDt);
    verify("unlocked movement retained",
           (lockProbe.player().position - freeStart).horizontal().length() > 0.05); // 102
    lockProbe.togglePlayerPositionLock();
    const Vec3 lockAnchor = lockProbe.player().position;
    verify("position lock captures and stops", lockProbe.playerPositionLocked()
        && lockProbe.player().velocity.lengthSquared() == 0.0);                  // 103
    lockProbe.setMoveInput(1.0, -1.0);
    const bool lockedJumpRejected = !lockProbe.jump();
    for (int tick = 0; tick < 120; ++tick) lockProbe.step(kFixedDt);
    verify("locked position invariant", samePosition(lockProbe.player().position,
                                                      lockAnchor)
        && lockProbe.player().velocity.lengthSquared() == 0.0
        && lockProbe.player().moveForward == 0.0
        && lockProbe.player().moveStrafe == 0.0);                                // 104
    verify("locked jump rejected", lockedJumpRejected);                          // 105
    lockProbe.player().yaw = 0.35;
    lockProbe.player().pitch = -0.12;
    lockProbe.step(kFixedDt);
    verify("locked camera remains free", lockProbe.player().yaw == 0.35
        && lockProbe.player().pitch == -0.12
        && samePosition(lockProbe.player().position, lockAnchor));                // 106

    PhysicsWorld lockedInteraction(78, 0);
    sleepAllBodies(lockedInteraction);
    lockedInteraction.togglePlayerPositionLock();
    const Vec3 interactionAnchor = lockedInteraction.player().position;
    const bool lockedPickup = lockedInteraction.pickupOrDrop();
    const int lockedThrownIndex = lockedInteraction.heldBody();
    verify("locked pickup remains active", lockedPickup && lockedThrownIndex >= 0); // 107
    bool heldConstraintActive = false;
    if (lockedPickup) {
        const Vec3 heldStart = lockedInteraction.bodies()[
            static_cast<std::size_t>(lockedThrownIndex)].position;
        lockedInteraction.player().yaw = 0.12;
        lockedInteraction.player().pitch = -0.22;
        for (int tick = 0; tick < 12; ++tick) lockedInteraction.step(kFixedDt);
        heldConstraintActive =
            (lockedInteraction.bodies()[static_cast<std::size_t>(lockedThrownIndex)]
                 .position - heldStart).length() > 0.01
            && lockedInteraction.heldBody() == lockedThrownIndex
            && samePosition(lockedInteraction.player().position, interactionAnchor)
            && lockedInteraction.player().velocity.lengthSquared() == 0.0;
    }
    verify("locked hold follows free camera", heldConstraintActive);              // 108
    const bool lockedThrow = lockedPickup && lockedInteraction.throwHeld();
    verify("locked throw without recoil", lockedThrow
        && lockedInteraction.bodies()[static_cast<std::size_t>(lockedThrownIndex)]
               .velocity.length() > 0.0
        && samePosition(lockedInteraction.player().position, interactionAnchor)
        && lockedInteraction.player().velocity.lengthSquared() == 0.0);          // 109

    PhysicsWorld lockedCollision(91, 0);
    sleepAllBodies(lockedCollision);
    lockedCollision.togglePlayerPositionLock();
    const Vec3 collisionAnchor = lockedCollision.player().position;
    RigidBody& incoming = lockedCollision.bodies().front();
    incoming.position = {10.0, 0.90, 8.60};
    incoming.previousPosition = incoming.position;
    incoming.velocity = {0.0, 0.0, 24.0};
    incoming.angularVelocity = {};
    incoming.force = {};
    incoming.torque = {};
    incoming.asleep = false;
    incoming.grounded = false;
    bool incomingDeflected = false;
    for (int tick = 0; tick < 24; ++tick) {
        lockedCollision.step(kFixedDt);
        incomingDeflected = incomingDeflected || incoming.velocity.z < 0.0;
    }
    verify("locked player is a static collider", incomingDeflected
        && samePosition(lockedCollision.player().position, collisionAnchor)
        && lockedCollision.player().velocity.lengthSquared() == 0.0);            // 110

    RigidBody& sleepingOverlap = lockedCollision.bodies()[1];
    sleepingOverlap.position = {10.0, 0.90, 10.10};
    sleepingOverlap.previousPosition = sleepingOverlap.position;
    sleepingOverlap.velocity = {};
    sleepingOverlap.angularVelocity = {};
    sleepingOverlap.force = {};
    sleepingOverlap.torque = {};
    sleepingOverlap.asleep = true;
    sleepingOverlap.grounded = false;
    const Vec3 sleepingStart = sleepingOverlap.position;
    lockedCollision.step(kFixedDt);
    verify("locked overlap wakes movable body", !sleepingOverlap.asleep
        && !samePosition(sleepingOverlap.position, sleepingStart)
        && samePosition(lockedCollision.player().position, collisionAnchor));     // 111

    lockProbe.togglePlayerPositionLock();
    const Vec3 unlockStart = lockProbe.player().position;
    lockProbe.setMoveInput(1.0, 0.0);
    for (int tick = 0; tick < 24; ++tick) lockProbe.step(kFixedDt);
    verify("unlock restores movement physics", !lockProbe.playerPositionLocked()
        && (lockProbe.player().position - unlockStart).horizontal().length() > 0.05); // 112
    lockProbe.togglePlayerPositionLock();
    lockProbe.reset();
    const Vec3 resetAnchor = lockProbe.player().position;
    verify("F5 clears position-lock mode",
           !lockProbe.playerPositionLocked()
        && samePosition(resetAnchor, Player{}.position)
        && samePosition(lockProbe.player().previousPosition,
                        Player{}.previousPosition)
        && lockProbe.player().velocity.lengthSquared() == 0.0);                  // 113

    PhysicsWorld midairLock(93, 0);
    sleepAllBodies(midairLock);
    const bool midairJumped = midairLock.jump();
    for (int tick = 0; tick < 12; ++tick) midairLock.step(kFixedDt);
    midairLock.togglePlayerPositionLock();
    const Vec3 midairAnchor = midairLock.player().position;
    for (int tick = 0; tick < 120; ++tick) midairLock.step(kFixedDt);
    const bool heldMidair = samePosition(midairLock.player().position, midairAnchor);
    midairLock.togglePlayerPositionLock();
    for (int tick = 0; tick < 300 && !midairLock.player().grounded; ++tick) {
        midairLock.step(kFixedDt);
    }
    verify("midair lock then gravity resume", midairJumped && midairAnchor.y > 0.0
        && heldMidair && midairLock.player().grounded
        && midairLock.player().position.y == 0.0);                                // 114

    PhysicsWorld lockedBulkOverlap(94, 0);
    sleepAllBodies(lockedBulkOverlap);
    lockedBulkOverlap.player().position = {86.0, 0.0, 44.0};
    lockedBulkOverlap.player().previousPosition = lockedBulkOverlap.player().position;
    lockedBulkOverlap.togglePlayerPositionLock();
    lockedBulkOverlap.step(kFixedDt);
    const auto lockedBulkAwake = std::count_if(
        lockedBulkOverlap.bodies().begin(), lockedBulkOverlap.bodies().end(),
        [&](const RigidBody& body) {
            return isBulkStock(lockedBulkOverlap.specs()[body.spec])
                && !body.asleep;
        });
    verify("locked overlap respects staged bulk wake cap",
           lockedBulkAwake > 0
        && lockedBulkAwake <= kMaximumNewBulkWakes);                              // 115

    // Exact full-reset and spatial-broadphase regression probes (116-121).
    PhysicsWorld resetRoundTrip(5'151, 0);
    const std::vector<RigidBody> initialBodies = resetRoundTrip.bodies();
    std::vector<std::uint16_t> initialPlushSpecs;
    for (const RigidBody& body : initialBodies) {
        if (resetRoundTrip.specs()[body.spec].category == "plush") {
            initialPlushSpecs.push_back(body.spec);
        }
    }
    sleepAllBodies(resetRoundTrip);
    resetRoundTrip.step(kFixedDt);
    const bool resetHeldSomething = resetRoundTrip.pickupOrDrop();
    resetRoundTrip.adjustGravity(1);
    resetRoundTrip.adjustFriction(1);
    resetRoundTrip.adjustThrowForce(1);
    resetRoundTrip.togglePlayerPositionLock();
    resetRoundTrip.player().position = {44.0, 1.5, 33.0};
    resetRoundTrip.player().previousPosition = {43.0, 1.0, 32.0};
    resetRoundTrip.bodies().back().position = {3.0, 7.0, 91.0};
    resetRoundTrip.bodies().back().orientation = {0.5, 0.5, 0.5, 0.5};
    resetRoundTrip.bodies().back().velocity = {4.0, 5.0, 6.0};
    resetRoundTrip.reset();

    const bool exactInitialTransforms = resetRoundTrip.bodies().size()
        == initialBodies.size() && [&] {
            for (std::size_t index = 0; index < initialBodies.size(); ++index) {
                const RigidBody& expected = initialBodies[index];
                const RigidBody& actual = resetRoundTrip.bodies()[index];
                if (actual.spec != expected.spec
                    || !sameVec(actual.position, expected.position)
                    || !sameVec(actual.previousPosition, expected.previousPosition)
                    || !sameQuat(actual.orientation, expected.orientation)
                    || !sameQuat(actual.previousOrientation,
                                 expected.previousOrientation)
                    || !sameVec(actual.colorOverride, expected.colorOverride)
                    || actual.cachedBoundingRadius != expected.cachedBoundingRadius
                    || actual.groupIndex != expected.groupIndex
                    || actual.grounded != expected.grounded
                    || actual.instanceLabel != expected.instanceLabel) {
                    return false;
                }
            }
            return true;
        }();
    verify("F5 restores every exact item transform", exactInitialTransforms);    // 116

    std::vector<std::uint16_t> resetPlushSpecs;
    for (const RigidBody& body : resetRoundTrip.bodies()) {
        if (resetRoundTrip.specs()[body.spec].category == "plush") {
            resetPlushSpecs.push_back(body.spec);
        }
    }
    verify("F5 deterministically restores plush selection",
           resetPlushSpecs == initialPlushSpecs);                                // 117
    verify("F5 restores clean all-awake body state", std::all_of(
        resetRoundTrip.bodies().begin(), resetRoundTrip.bodies().end(),
        [](const RigidBody& body) {
            return !body.asleep && body.pristine && !body.held
                && !body.stuckSurface && body.stuckTo < 0
                && !body.massCarriedByHost
                && body.velocity.lengthSquared() == 0.0
                && body.angularVelocity.lengthSquared() == 0.0
                && body.force.lengthSquared() == 0.0
                && body.torque.lengthSquared() == 0.0
                && body.sleepTime == 0.0 && body.impactCooldown == 0.0
                && body.lastImpulse == 0.0 && body.attachedPayloadMass == 0.0;
        }));                                                                     // 118
    verify("F5 clears transient world state",
           resetHeldSomething
        && resetRoundTrip.heldBody() == -1
        && resetRoundTrip.simulationTime() == 0.0
        && resetRoundTrip.gravity() == kEarthGravity
        && resetRoundTrip.roomFriction() == kDefaultRoomFriction
        && resetRoundTrip.throwForce() == kThrowForceMin
        && !resetRoundTrip.playerPositionLocked()
        && sameVec(resetRoundTrip.player().position, Player{}.position)
        && sameVec(resetRoundTrip.player().previousPosition,
                   Player{}.previousPosition));                                 // 119

    PhysicsWorld spatialProbe(6'201, 0);
    sleepAllBodies(spatialProbe);
    RigidBody& spatialMover = spatialProbe.bodies()[0];
    RigidBody& spatialSleeper = spatialProbe.bodies()[1];
    const Real spatialSeparation = spatialMover.cachedBoundingRadius
                                 + spatialSleeper.cachedBoundingRadius + 0.04;
    spatialMover.position = {40.0, 2.0, 40.0};
    spatialMover.previousPosition = spatialMover.position;
    spatialMover.velocity = {12.0, 0.0, 0.0};
    spatialMover.asleep = false;
    spatialMover.grounded = false;
    spatialMover.pristine = false;
    spatialSleeper.position = {40.0 + spatialSeparation, 2.0, 40.0};
    spatialSleeper.previousPosition = spatialSleeper.position;
    spatialSleeper.grounded = false;
    spatialSleeper.pristine = false;
    spatialProbe.step(kFixedDt);
    verify("spatial broadphase active-small against sleeping-large",
           spatialProbe.broadphaseCandidates() >= 1
        && !spatialSleeper.asleep
        && spatialSleeper.velocity.x > 0.0
        && spatialMover.velocity.x < 12.0);                                     // 120

    PhysicsWorld ccdProbe(6'202, 0);
    sleepAllBodies(ccdProbe);
    const auto bodyWithKey = [&](PhysicsWorld& probe,
                                 std::string_view key) -> RigidBody& {
        const auto iterator = std::find_if(
            probe.bodies().begin(), probe.bodies().end(), [&](const RigidBody& body) {
                return probe.specs()[body.spec].key == key;
            });
        if (iterator == probe.bodies().end()) {
            throw std::logic_error("self-check missing body: " + std::string(key));
        }
        return *iterator;
    };
    RigidBody& ccdMover = bodyWithKey(ccdProbe, "steel");
    RigidBody& ccdTarget = bodyWithKey(ccdProbe, "concrete");
    ccdMover.position = {40.0, 2.0, 20.0};
    ccdMover.previousPosition = ccdMover.position;
    ccdMover.velocity = {0.0, 0.0, 900.0};
    ccdMover.asleep = false;
    ccdMover.grounded = false;
    ccdMover.pristine = false;
    ccdTarget.position = {40.0, 2.0, 25.0};
    ccdTarget.previousPosition = ccdTarget.position;
    ccdTarget.grounded = false;
    ccdTarget.pristine = false;
    ccdProbe.step(kFixedDt);
    verify("spatial broadphase preserves high-speed CCD",
           ccdProbe.broadphaseCandidates() >= 1
        && !ccdTarget.asleep
        && ccdTarget.velocity.z > 0.0
        && ccdMover.velocity.z < 900.0
        && ccdMover.position.z < ccdTarget.position.z
             + ccdMover.cachedBoundingRadius + ccdTarget.cachedBoundingRadius); // 121

    // Galileo vacuum-drop experiment and bounded echo history (122-126).
    PhysicsWorld galileoProbe(6'203, 0);
    sleepAllBodies(galileoProbe);
    galileoProbe.togglePlayerPositionLock();
    const std::size_t galileoPopulation = galileoProbe.bodies().size();
    galileoProbe.startGalileoExperiment();
    const GalileoBodyIds galileoIds = galileoProbe.galileoBodyIds();
    const bool galileoIdsValid = galileoIds.steel >= 0
        && galileoIds.concrete >= 0
        && galileoIds.steel != galileoIds.concrete
        && static_cast<std::size_t>(galileoIds.steel)
             < galileoProbe.bodies().size()
        && static_cast<std::size_t>(galileoIds.concrete)
             < galileoProbe.bodies().size();
    const RigidBody& galileoSteel = galileoProbe.bodies()[
        static_cast<std::size_t>(galileoIds.steel)];
    const RigidBody& galileoConcrete = galileoProbe.bodies()[
        static_cast<std::size_t>(galileoIds.concrete)];
    const BodySpec& galileoSteelSpec = galileoProbe.specs()[galileoSteel.spec];
    const BodySpec& galileoConcreteSpec =
        galileoProbe.specs()[galileoConcrete.spec];
    const auto initialGalileoTrail = galileoProbe.galileoCurrentTrail();
    verify("Galileo setup reuses equal-diameter calibration bodies",
           galileoIdsValid
        && galileoProbe.bodies().size() == galileoPopulation
        && galileoPopulation == static_cast<std::size_t>(kExpectedBodyCount)
        && galileoSteelSpec.key == "steel"
        && galileoConcreteSpec.key == "concrete"
        && galileoSteelSpec.mass != galileoConcreteSpec.mass
        && galileoSteelSpec.diameter == galileoConcreteSpec.diameter
        && sameVec(galileoSteel.position, kGalileoSteelDropPosition)
        && sameVec(galileoConcrete.position, kGalileoConcreteDropPosition)
        && galileoSteel.position.y == galileoConcrete.position.y
        && sameVec(galileoSteel.velocity, {})
        && sameVec(galileoConcrete.velocity, {})
        && sameVec(galileoSteel.angularVelocity, {})
        && sameVec(galileoConcrete.angularVelocity, {})
        && sameQuat(galileoSteel.orientation, {})
        && sameQuat(galileoConcrete.orientation, {})
        && !galileoSteel.asleep && !galileoConcrete.asleep
        && !galileoSteel.grounded && !galileoConcrete.grounded
        && !galileoProbe.playerPositionLocked()
        && galileoProbe.galileoRunning()
        && galileoProbe.galileoStatus() == "Galileo vacuum drop running"
        && galileoProbe.galileoElapsed() == 0.0
        && !galileoProbe.galileoSteelLandingTime()
        && !galileoProbe.galileoConcreteLandingTime()
        && initialGalileoTrail.size() == 1
        && initialGalileoTrail.front().elapsed == 0.0
        && sameVec(initialGalileoTrail.front().steel,
                   kGalileoSteelDropPosition)
        && sameVec(initialGalileoTrail.front().concrete,
                   kGalileoConcreteDropPosition)
        && galileoProbe.galileoPreviousTrail().empty());                        // 122

    bool equalVacuumTrajectory = true;
    for (int tick = 0; tick < 60; ++tick) {
        galileoProbe.step(kFixedDt);
        equalVacuumTrajectory = equalVacuumTrajectory
            && near(galileoSteel.position.y, galileoConcrete.position.y,
                    1.0e-12)
            && near(galileoSteel.velocity.y, galileoConcrete.velocity.y,
                    1.0e-12);
    }
    verify("Galileo unequal masses share vacuum acceleration",
           equalVacuumTrajectory
        && galileoSteel.position.y < kGalileoSteelDropPosition.y
        && near(galileoSteel.velocity.y / galileoProbe.galileoElapsed(),
                -galileoProbe.gravity(), 1.0e-10)
        && galileoProbe.galileoCurrentTrail().size() == 61);                    // 123

    for (int tick = 0; tick < 300 && galileoProbe.galileoRunning(); ++tick) {
        galileoProbe.step(kFixedDt);
    }
    const std::size_t completedTrailSize =
        galileoProbe.galileoCurrentTrail().size();
    for (int tick = 0; tick < 60; ++tick) galileoProbe.step(kFixedDt);
    verify("Galileo trail is bounded and stops at first paired landing",
           galileoProbe.galileoState() == GalileoExperimentState::Complete
        && galileoProbe.galileoStatus() == "Galileo vacuum drop complete"
        && galileoProbe.galileoSteelLandingTime()
        && galileoProbe.galileoConcreteLandingTime()
        && near(*galileoProbe.galileoSteelLandingTime(),
                *galileoProbe.galileoConcreteLandingTime(), 1.0e-12)
        && completedTrailSize > 61
        && completedTrailSize <= kGalileoMaxTrailSamples
        && galileoProbe.galileoCurrentTrail().size() == completedTrailSize);    // 124

    const std::vector<GalileoTrailSample> completedTrail(
        galileoProbe.galileoCurrentTrail().begin(),
        galileoProbe.galileoCurrentTrail().end());
    galileoProbe.startGalileoExperiment();
    const auto echoTrail = galileoProbe.galileoPreviousTrail();
    bool exactEcho = echoTrail.size() == completedTrail.size();
    for (std::size_t index = 0; exactEcho && index < echoTrail.size(); ++index) {
        exactEcho = echoTrail[index].elapsed == completedTrail[index].elapsed
            && sameVec(echoTrail[index].steel, completedTrail[index].steel)
            && sameVec(echoTrail[index].concrete, completedTrail[index].concrete);
    }
    verify("Galileo restart preserves previous-run echo",
           exactEcho
        && galileoProbe.galileoRunning()
        && galileoProbe.galileoElapsed() == 0.0
        && galileoProbe.galileoCurrentTrail().size() == 1
        && sameVec(galileoSteel.position, kGalileoSteelDropPosition)
        && sameVec(galileoConcrete.position, kGalileoConcreteDropPosition)
        && !galileoProbe.galileoSteelLandingTime()
        && !galileoProbe.galileoConcreteLandingTime());                         // 125

    galileoProbe.step(kFixedDt);
    galileoProbe.reset();
    const GalileoBodyIds resetGalileoIds = galileoProbe.galileoBodyIds();
    verify("full reset clears Galileo experiment and echo state",
           galileoProbe.bodies().size()
               == static_cast<std::size_t>(kExpectedBodyCount)
        && galileoProbe.galileoState() == GalileoExperimentState::Inactive
        && !galileoProbe.galileoRunning()
        && galileoProbe.galileoStatus() == "Galileo experiment ready"
        && resetGalileoIds.steel == -1 && resetGalileoIds.concrete == -1
        && galileoProbe.galileoElapsed() == 0.0
        && !galileoProbe.galileoSteelLandingTime()
        && !galileoProbe.galileoConcreteLandingTime()
        && galileoProbe.galileoCurrentTrail().empty()
        && galileoProbe.galileoPreviousTrail().empty());                        // 126

    PhysicsWorld frameImpactProbe(6'204, 0);
    sleepAllBodies(frameImpactProbe);
    RigidBody& frameImpactBody = bodyWithKey(frameImpactProbe, "steel");
    const auto stageRoomImpact = [&] {
        const Real radius = frameImpactBody.cachedBoundingRadius;
        frameImpactBody.position = {50.0, radius + 0.02, 50.0};
        frameImpactBody.previousPosition = frameImpactBody.position;
        frameImpactBody.velocity = {0.0, -20.0, 0.0};
        frameImpactBody.angularVelocity = {};
        frameImpactBody.force = {};
        frameImpactBody.torque = {};
        frameImpactBody.asleep = false;
        frameImpactBody.grounded = false;
        frameImpactBody.pristine = false;
        frameImpactBody.impactCooldown = 0.0;
    };
    frameImpactProbe.beginFrameEvents();
    stageRoomImpact();
    frameImpactProbe.step(kFixedDt);
    const std::size_t firstImpactCount = frameImpactProbe.impacts().size();
    stageRoomImpact();
    frameImpactProbe.step(kFixedDt);
    const std::size_t aggregatedImpactCount = frameImpactProbe.impacts().size();
    frameImpactProbe.endFrameEvents();
    stageRoomImpact();
    frameImpactProbe.step(kFixedDt);
    const std::size_t standaloneImpactCount = frameImpactProbe.impacts().size();
    frameImpactProbe.beginFrameEvents();
    frameImpactProbe.reset();
    verify("render-frame impact collection retains every fixed substep",
           firstImpactCount == 1U
        && aggregatedImpactCount == 2U
        && standaloneImpactCount == 1U
        && frameImpactProbe.impacts().empty());                                 // 127

    // Gravity-proof falling-target intercept and paired echo history
    // (128-134). The marble and seed-selected plush are existing scene bodies;
    // only their authoritative state is staged for the experiment.
    PhysicsWorld interceptProbe(6'205, 0);
    sleepAllBodies(interceptProbe);
    interceptProbe.togglePlayerPositionLock();
    const std::size_t interceptPopulation = interceptProbe.bodies().size();
    interceptProbe.startGravityInterceptExperiment();
    const GravityInterceptBodyIds interceptIds =
        interceptProbe.gravityInterceptBodyIds();
    const bool interceptIdsValid = interceptIds.projectile >= 0
        && interceptIds.target >= 0
        && interceptIds.projectile != interceptIds.target
        && static_cast<std::size_t>(interceptIds.projectile)
             < interceptProbe.bodies().size()
        && static_cast<std::size_t>(interceptIds.target)
             < interceptProbe.bodies().size();
    const RigidBody& interceptProjectile = interceptProbe.bodies()[
        static_cast<std::size_t>(interceptIds.projectile)];
    const RigidBody& interceptTarget = interceptProbe.bodies()[
        static_cast<std::size_t>(interceptIds.target)];
    const BodySpec& interceptProjectileSpec =
        interceptProbe.specs()[interceptProjectile.spec];
    const BodySpec& interceptTargetSpec =
        interceptProbe.specs()[interceptTarget.spec];
    const Vec3 interceptDelta = kGravityInterceptTargetPosition
                              - kGravityInterceptProjectilePosition;
    const Vec3 interceptLaunchVelocity = interceptDelta.normalized()
                                       * kGravityInterceptLaunchSpeed;
    const Real interceptRadii = interceptProjectileSpec.radius()
                              + interceptTargetSpec.radius();
    const Real initialInterceptGap = interceptDelta.length() - interceptRadii;
    const auto initialInterceptTrail =
        interceptProbe.gravityInterceptCurrentTrail();
    verify("gravity intercept stages existing marble and plush bodies",
           interceptIdsValid
        && interceptProbe.bodies().size() == interceptPopulation
        && interceptPopulation == static_cast<std::size_t>(kExpectedBodyCount)
        && interceptProjectileSpec.key == "ceramic_marble"
        && interceptTargetSpec.category == "plush"
        && interceptProjectileSpec.addedMass == 0.0
        && interceptTargetSpec.addedMass == 0.0
        && interceptTargetSpec.linearDamping > 0.0
        && sameVec(interceptProjectile.position,
                   kGravityInterceptProjectilePosition)
        && sameVec(interceptTarget.position, kGravityInterceptTargetPosition)
        && sameVec(interceptProjectile.velocity, interceptLaunchVelocity)
        && sameVec(interceptTarget.velocity, {})
        && !interceptProjectile.asleep && !interceptTarget.asleep
        && !interceptProjectile.grounded && !interceptTarget.grounded
        && !interceptProbe.playerPositionLocked()
        && interceptProbe.gravityInterceptRunning()
        && interceptProbe.gravityInterceptStatus()
             == "Gravity-proof trick shot running"
        && interceptProbe.gravityInterceptElapsed() == 0.0
        && !interceptProbe.gravityInterceptHitTime()
        && !interceptProbe.gravityInterceptHitPoint()
        && interceptProbe.gravityInterceptClosestGap()
        && near(*interceptProbe.gravityInterceptClosestGap(),
                initialInterceptGap, 1.0e-12)
        && initialInterceptTrail.size() == 1U
        && initialInterceptTrail.front().elapsed == 0.0
        && sameVec(initialInterceptTrail.front().projectile,
                   kGravityInterceptProjectilePosition)
        && sameVec(initialInterceptTrail.front().target,
                   kGravityInterceptTargetPosition)
        && interceptProbe.gravityInterceptPreviousTrail().empty());            // 128

    bool relativeFlightInvariant = true;
    for (int tick = 0; tick < 40; ++tick) {
        interceptProbe.step(kFixedDt);
        const Vec3 expectedRelative = interceptDelta
            - interceptLaunchVelocity * interceptProbe.gravityInterceptElapsed();
        const Vec3 actualRelative = interceptTarget.position
                                  - interceptProjectile.position;
        const Vec3 relativeVelocity = interceptTarget.velocity
                                    - interceptProjectile.velocity;
        relativeFlightInvariant = relativeFlightInvariant
            && near(actualRelative.x, expectedRelative.x, 1.0e-10)
            && near(actualRelative.y, expectedRelative.y, 1.0e-10)
            && near(actualRelative.z, expectedRelative.z, 1.0e-10)
            && near(relativeVelocity.x, -interceptLaunchVelocity.x, 1.0e-10)
            && near(relativeVelocity.y, -interceptLaunchVelocity.y, 1.0e-10)
            && near(relativeVelocity.z, -interceptLaunchVelocity.z, 1.0e-10);
    }
    verify("gravity intercept local vacuum preserves relative firing line",
           relativeFlightInvariant
        && interceptProbe.gravityInterceptRunning()
        && interceptProbe.gravityInterceptCurrentTrail().size() == 41U);       // 129

    for (int tick = 0; tick < 120
         && interceptProbe.gravityInterceptRunning(); ++tick) {
        interceptProbe.step(kFixedDt);
    }
    const double analyticInterceptTime =
        initialInterceptGap / kGravityInterceptLaunchSpeed;
    const std::size_t completedInterceptTrailSize =
        interceptProbe.gravityInterceptCurrentTrail().size();
    for (int tick = 0; tick < 20; ++tick) interceptProbe.step(kFixedDt);
    verify("gravity intercept records authoritative CCD contact",
           interceptProbe.gravityInterceptState() == GravityInterceptState::Hit
        && interceptProbe.gravityInterceptStatus()
             == "Gravity-proof trick shot hit"
        && interceptProbe.gravityInterceptHitTime()
        && *interceptProbe.gravityInterceptHitTime() >= analyticInterceptTime
        && *interceptProbe.gravityInterceptHitTime()
             <= analyticInterceptTime + kFixedDt + 1.0e-12
        && interceptProbe.gravityInterceptHitPoint()
        && interceptProbe.gravityInterceptHitPoint()->finite()
        && interceptProbe.gravityInterceptClosestGap()
        && *interceptProbe.gravityInterceptClosestGap() == 0.0
        && completedInterceptTrailSize > 41U
        && completedInterceptTrailSize <= kGravityInterceptMaxTrailSamples
        && interceptProbe.gravityInterceptCurrentTrail().size()
             == completedInterceptTrailSize);                                  // 130

    const std::vector<GravityInterceptTrailSample> completedInterceptTrail(
        interceptProbe.gravityInterceptCurrentTrail().begin(),
        interceptProbe.gravityInterceptCurrentTrail().end());
    int attachedGooId = -1;
    for (std::size_t index = 0; index < interceptProbe.bodies().size(); ++index) {
        if (interceptProbe.specs()[interceptProbe.bodies()[index].spec].key
            == "sticky_goo") {
            attachedGooId = static_cast<int>(index);
            break;
        }
    }
    if (attachedGooId < 0) {
        throw std::logic_error("gravity intercept self-check missing goo");
    }
    RigidBody& attachedGoo = interceptProbe.bodies()[
        static_cast<std::size_t>(attachedGooId)];
    RigidBody& payloadTarget = interceptProbe.bodies()[
        static_cast<std::size_t>(interceptIds.target)];
    attachedGoo.stuckTo = interceptIds.target;
    attachedGoo.massCarriedByHost = true;
    payloadTarget.attachedPayloadMass =
        interceptProbe.specs()[attachedGoo.spec].mass;
    const bool heldBeforeInterceptRestart = interceptProbe.pickupOrDrop();
    const int releasedHeldId = interceptProbe.heldBody();
    interceptProbe.startGravityInterceptExperiment();
    const auto interceptEcho = interceptProbe.gravityInterceptPreviousTrail();
    bool exactInterceptEcho = interceptEcho.size()
                           == completedInterceptTrail.size();
    for (std::size_t index = 0;
         exactInterceptEcho && index < interceptEcho.size(); ++index) {
        exactInterceptEcho =
            interceptEcho[index].elapsed
                == completedInterceptTrail[index].elapsed
            && sameVec(interceptEcho[index].projectile,
                       completedInterceptTrail[index].projectile)
            && sameVec(interceptEcho[index].target,
                       completedInterceptTrail[index].target);
    }
    verify("gravity intercept restart preserves echo and clears carried state",
           heldBeforeInterceptRestart && releasedHeldId >= 0
        && interceptProbe.heldBody() == -1
        && !interceptProbe.bodies()[static_cast<std::size_t>(releasedHeldId)].held
        && attachedGoo.stuckTo == -1
        && !attachedGoo.massCarriedByHost
        && payloadTarget.attachedPayloadMass == 0.0
        && near(interceptProbe.dynamicMass(payloadTarget),
                interceptTargetSpec.mass, 1.0e-12)
        && exactInterceptEcho
        && interceptProbe.gravityInterceptRunning()
        && interceptProbe.gravityInterceptElapsed() == 0.0
        && interceptProbe.gravityInterceptCurrentTrail().size() == 1U
        && !interceptProbe.gravityInterceptHitTime()
        && !interceptProbe.gravityInterceptHitPoint());                        // 131

    interceptProbe.step(kFixedDt);
    interceptProbe.reset();
    const GravityInterceptBodyIds resetInterceptIds =
        interceptProbe.gravityInterceptBodyIds();
    verify("full reset clears gravity intercept and echo state",
           interceptProbe.bodies().size()
               == static_cast<std::size_t>(kExpectedBodyCount)
        && interceptProbe.gravityInterceptState()
             == GravityInterceptState::Inactive
        && !interceptProbe.gravityInterceptRunning()
        && interceptProbe.gravityInterceptStatus() == "Gravity intercept ready"
        && resetInterceptIds.projectile == -1 && resetInterceptIds.target == -1
        && interceptProbe.gravityInterceptElapsed() == 0.0
        && !interceptProbe.gravityInterceptHitTime()
        && !interceptProbe.gravityInterceptHitPoint()
        && !interceptProbe.gravityInterceptClosestGap()
        && interceptProbe.gravityInterceptCurrentTrail().empty()
        && interceptProbe.gravityInterceptPreviousTrail().empty());            // 132

    PhysicsWorld lowGravityIntercept(6'206, 0);
    PhysicsWorld highGravityIntercept(6'206, 0);
    for (int adjustment = 0; adjustment < 256
         && lowGravityIntercept.gravity() > kGravityMin; ++adjustment) {
        lowGravityIntercept.adjustGravity(-1);
    }
    for (int adjustment = 0; adjustment < 256
         && highGravityIntercept.gravity() < kGravityMax; ++adjustment) {
        highGravityIntercept.adjustGravity(1);
    }
    sleepAllBodies(lowGravityIntercept);
    sleepAllBodies(highGravityIntercept);
    lowGravityIntercept.startGravityInterceptExperiment();
    highGravityIntercept.startGravityInterceptExperiment();
    for (int tick = 0; tick < 120
         && (lowGravityIntercept.gravityInterceptRunning()
             || highGravityIntercept.gravityInterceptRunning()); ++tick) {
        if (lowGravityIntercept.gravityInterceptRunning()) {
            lowGravityIntercept.step(kFixedDt);
        }
        if (highGravityIntercept.gravityInterceptRunning()) {
            highGravityIntercept.step(kFixedDt);
        }
    }
    verify("gravity extremes bend both paths without changing intercept time",
           lowGravityIntercept.gravity() == kGravityMin
        && highGravityIntercept.gravity() == kGravityMax
        && lowGravityIntercept.gravityInterceptState()
             == GravityInterceptState::Hit
        && highGravityIntercept.gravityInterceptState()
             == GravityInterceptState::Hit
        && lowGravityIntercept.gravityInterceptHitTime()
        && highGravityIntercept.gravityInterceptHitTime()
        && near(*lowGravityIntercept.gravityInterceptHitTime(),
                *highGravityIntercept.gravityInterceptHitTime(), 1.0e-10)
        && lowGravityIntercept.gravityInterceptHitPoint()
        && highGravityIntercept.gravityInterceptHitPoint()
        && lowGravityIntercept.gravityInterceptHitPoint()->y
             - highGravityIntercept.gravityInterceptHitPoint()->y > 6.0);      // 133

    PhysicsWorld scalarIntercept(6'207, 0, "scalar");
    PhysicsWorld helperIntercept(6'207, 3, "maximum");
    sleepAllBodies(scalarIntercept);
    sleepAllBodies(helperIntercept);
    scalarIntercept.startGravityInterceptExperiment();
    helperIntercept.startGravityInterceptExperiment();
    for (int tick = 0; tick < 120
         && (scalarIntercept.gravityInterceptRunning()
             || helperIntercept.gravityInterceptRunning()); ++tick) {
        if (scalarIntercept.gravityInterceptRunning()) {
            scalarIntercept.step(kFixedDt);
        }
        if (helperIntercept.gravityInterceptRunning()) {
            helperIntercept.step(kFixedDt);
        }
    }
    const auto scalarInterceptTrail =
        scalarIntercept.gravityInterceptCurrentTrail();
    const auto helperInterceptTrail =
        helperIntercept.gravityInterceptCurrentTrail();
    bool exactHelperIntercept = scalarInterceptTrail.size()
                             == helperInterceptTrail.size();
    for (std::size_t index = 0;
         exactHelperIntercept && index < scalarInterceptTrail.size(); ++index) {
        exactHelperIntercept =
            scalarInterceptTrail[index].elapsed
                == helperInterceptTrail[index].elapsed
            && sameVec(scalarInterceptTrail[index].projectile,
                       helperInterceptTrail[index].projectile)
            && sameVec(scalarInterceptTrail[index].target,
                       helperInterceptTrail[index].target);
    }
    verify("gravity intercept is deterministic across helper backends",
           scalarIntercept.gravityInterceptState()
               == GravityInterceptState::Hit
        && helperIntercept.gravityInterceptState()
               == GravityInterceptState::Hit
        && scalarIntercept.gravityInterceptHitTime()
        && helperIntercept.gravityInterceptHitTime()
        && *scalarIntercept.gravityInterceptHitTime()
             == *helperIntercept.gravityInterceptHitTime()
        && scalarIntercept.gravityInterceptHitPoint()
        && helperIntercept.gravityInterceptHitPoint()
        && sameVec(*scalarIntercept.gravityInterceptHitPoint(),
                   *helperIntercept.gravityInterceptHitPoint())
        && scalarIntercept.gravityInterceptElapsed()
             == helperIntercept.gravityInterceptElapsed()
        && exactHelperIntercept);                                               // 134

    const auto stageEchoPulse = [&](PhysicsWorld& probe) {
        sleepAllBodies(probe);
        probe.player().position = {40.0, 3.24, 32.0};
        probe.player().previousPosition = probe.player().position;
        probe.player().velocity = {};
        probe.player().yaw = 0.0;
        probe.player().pitch = 0.0;
        probe.player().grounded = false;
        std::array<int, 6> ids{};
        ids.fill(-1);
        std::size_t found = 0U;
        for (std::size_t index = 0;
             index < probe.bodies().size() && found < ids.size(); ++index) {
            if (probe.specs()[probe.bodies()[index].spec].key
                == "ceramic_marble") {
                ids[found++] = static_cast<int>(index);
            }
        }
        if (found != ids.size()) {
            throw std::logic_error("echo-pulse self-check missing marbles");
        }
        constexpr std::array<Vec3, 6> positions{
            Vec3{40.0, 5.0, 37.0}, Vec3{42.0, 5.0, 37.0},
            Vec3{47.5, 5.0, 37.0}, Vec3{48.02, 5.0, 37.0},
            Vec3{48.04, 5.0, 37.0}, Vec3{40.5, 5.0, 37.0},
        };
        for (std::size_t slot = 0; slot < ids.size(); ++slot) {
            RigidBody& body = probe.bodies()[static_cast<std::size_t>(ids[slot])];
            body.position = positions[slot];
            body.previousPosition = body.position;
            body.velocity = {};
            body.angularVelocity = {};
            body.force = {};
            body.torque = {};
            body.asleep = true;
            body.grounded = false;
            body.pristine = false;
            body.held = false;
        }
        probe.bodies()[static_cast<std::size_t>(ids[5])].held = true;
        return ids;
    };

    PhysicsWorld pulseProbe(7'301, 0);
    const auto pulseIds = stageEchoPulse(pulseProbe);
    const Vec3 pulseSource = pulseProbe.player().eye();
    const Vec3 playerBeforePulse = pulseProbe.player().velocity;
    const bool emittedPulse = pulseProbe.emitEchoPulse();
    const auto firstPulse = pulseProbe.lastEchoPulseEvent();
    const auto pulseBodies = pulseProbe.lastEchoPulseBodies();
    Real summedPulseImpulse = 0.0;
    for (const EchoPulseBodyEvent& bodyEvent : pulseBodies) {
        summedPulseImpulse += bodyEvent.impulse.length();
    }
    verify("echo pulse targets the crosshair body with authoritative telemetry",
           emittedPulse && firstPulse
        && firstPulse->serial == 1U
        && sameVec(firstPulse->source, pulseSource)
        && sameVec(firstPulse->origin,
                   pulseProbe.bodies()[static_cast<std::size_t>(pulseIds[0])]
                       .position)
        && sameVec(firstPulse->direction, {0.0, 0.0, 1.0})
        && firstPulse->simulationTime == 0.0
        && firstPulse->radius == kEchoPulseRadius
        && firstPulse->targetBody == pulseIds[0]
        && firstPulse->affectedBodyCount == 4U
        && pulseBodies.size() == 4U
        && near(firstPulse->totalDeliveredImpulse, summedPulseImpulse, 1.0e-12)); // 135

    const auto pulseBodyEvent = [&](int body) {
        return std::find_if(pulseBodies.begin(), pulseBodies.end(),
            [body](const EchoPulseBodyEvent& event) {
                return event.body == static_cast<std::uint16_t>(body);
            });
    };
    const auto targetPulseEvent = pulseBodyEvent(pulseIds[0]);
    const auto nearPulseEvent = pulseBodyEvent(pulseIds[1]);
    const auto farPulseEvent = pulseBodyEvent(pulseIds[2]);
    const auto radiusAwareEvent = pulseBodyEvent(pulseIds[3]);
    const auto outsidePulseEvent = pulseBodyEvent(pulseIds[4]);
    const Vec3 firstPulseOrigin = firstPulse ? firstPulse->origin : Vec3{};
    verify("echo pulse uses body-radius smooth falloff and overlap fallback",
           targetPulseEvent != pulseBodies.end()
        && nearPulseEvent != pulseBodies.end()
        && farPulseEvent != pulseBodies.end()
        && radiusAwareEvent != pulseBodies.end()
        && outsidePulseEvent == pulseBodies.end()
        && targetPulseEvent->surfaceDistance == 0.0
        && radiusAwareEvent->surfaceDistance < kEchoPulseRadius
        && (radiusAwareEvent->position - firstPulseOrigin).length()
             > kEchoPulseRadius
        && targetPulseEvent->impulse.z > 0.0
        && targetPulseEvent->impulse.x == 0.0
        && targetPulseEvent->impulse.y == 0.0
        && targetPulseEvent->impulse.length() > nearPulseEvent->impulse.length()
        && nearPulseEvent->impulse.length() > farPulseEvent->impulse.length()
        && farPulseEvent->impulse.length()
             > radiusAwareEvent->impulse.length());                            // 136
    verify("echo pulse excludes held bodies and applies bounded player recoil",
           pulseBodyEvent(pulseIds[5]) == pulseBodies.end()
        && pulseProbe.bodies()[static_cast<std::size_t>(pulseIds[5])]
               .velocity.lengthSquared() == 0.0
        && pulseProbe.bodies()[static_cast<std::size_t>(pulseIds[5])].asleep
        && pulseProbe.player().velocity.z < playerBeforePulse.z
        && (pulseProbe.player().velocity - playerBeforePulse).length() <= 0.71); // 137

    const bool rejectedDuringCooldown = !pulseProbe.emitEchoPulse();
    verify("echo pulse cooldown rejects duplicate simulation-time input",
           rejectedDuringCooldown && pulseProbe.lastEchoPulseEvent()
        && pulseProbe.lastEchoPulseEvent()->serial == 1U
        && near(pulseProbe.echoPulseCooldownRemaining(), kEchoPulseCooldown,
                1.0e-12)
        && !pulseProbe.echoPulseReady()
        && pulseProbe.echoPulseStatus() == "Echo pulse recharging");           // 138
    int rechargeTicks = 0;
    while (!pulseProbe.echoPulseReady() && rechargeTicks < 100) {
        pulseProbe.step(kFixedDt);
        ++rechargeTicks;
    }
    const bool emittedAfterCooldown = pulseProbe.emitEchoPulse();
    verify("echo pulse recharges only as fixed simulation time advances",
           rechargeTicks == 78 && emittedAfterCooldown
        && pulseProbe.lastEchoPulseEvent()
        && pulseProbe.lastEchoPulseEvent()->serial == 2U
        && near(pulseProbe.echoPulseCooldownRemaining(), kEchoPulseCooldown,
                1.0e-12));                                                     // 139

    pulseProbe.reset();
    verify("full reset clears echo pulse state",
           !pulseProbe.lastEchoPulseEvent()
        && pulseProbe.lastEchoPulseBodies().empty()
        && pulseProbe.echoPulseCooldownRemaining() == 0.0
        && pulseProbe.echoPulseReady()
        && pulseProbe.echoPulseStatus() == "Echo pulse ready");                // 140

    PhysicsWorld fallbackPulse(7'302, 0);
    sleepAllBodies(fallbackPulse);
    for (RigidBody& body : fallbackPulse.bodies()) {
        body.position = {90.0, 9.0, 90.0};
        body.previousPosition = body.position;
    }
    fallbackPulse.player().position = {40.0, 3.24, 32.0};
    fallbackPulse.player().previousPosition = fallbackPulse.player().position;
    fallbackPulse.player().yaw = 0.0;
    fallbackPulse.player().pitch = 0.0;
    const Vec3 fallbackSource = fallbackPulse.player().eye();
    const bool emittedFallback = fallbackPulse.emitEchoPulse();
    const auto fallbackEvent = fallbackPulse.lastEchoPulseEvent();
    verify("echo pulse uses the seven-metre free-space fallback",
           emittedFallback && fallbackEvent
        && fallbackEvent->targetBody == -1
        && sameVec(fallbackEvent->source, fallbackSource)
        && sameVec(fallbackEvent->origin,
                   fallbackSource + Vec3{0.0, 0.0,
                                         kEchoPulseFallbackDistance})
        && fallbackEvent->affectedBodyCount == 0U
        && fallbackEvent->totalDeliveredImpulse == 0.0);                       // 141

    PhysicsWorld lockedPulse(7'303, 0);
    const auto lockedPulseIds = stageEchoPulse(lockedPulse);
    lockedPulse.togglePlayerPositionLock();
    const Vec3 lockedPulseAnchor = lockedPulse.player().position;
    const bool emittedLockedPulse = lockedPulse.emitEchoPulse();
    verify("position lock suppresses echo pulse recoil without suppressing pulse",
           emittedLockedPulse && lockedPulse.playerPositionLocked()
        && sameVec(lockedPulse.player().position, lockedPulseAnchor)
        && lockedPulse.player().velocity.lengthSquared() == 0.0
        && lockedPulse.lastEchoPulseEvent()
        && lockedPulse.lastEchoPulseEvent()->affectedBodyCount == 4U
        && lockedPulse.bodies()[static_cast<std::size_t>(lockedPulseIds[0])]
               .velocity.lengthSquared() > 0.0);                               // 142

    PhysicsWorld scalarPulse(7'304, 0, "scalar");
    PhysicsWorld helperPulse(7'304, 3, "maximum");
    const auto scalarPulseIds = stageEchoPulse(scalarPulse);
    const auto helperPulseIds = stageEchoPulse(helperPulse);
    const bool scalarPulseEmitted = scalarPulse.emitEchoPulse();
    const bool helperPulseEmitted = helperPulse.emitEchoPulse();
    const auto scalarPulseEvent = scalarPulse.lastEchoPulseEvent();
    const auto helperPulseEvent = helperPulse.lastEchoPulseEvent();
    const auto scalarPulseBodies = scalarPulse.lastEchoPulseBodies();
    const auto helperPulseBodies = helperPulse.lastEchoPulseBodies();
    bool exactPulseEvents = scalarPulseBodies.size() == helperPulseBodies.size();
    for (std::size_t index = 0;
         exactPulseEvents && index < scalarPulseBodies.size(); ++index) {
        exactPulseEvents = scalarPulseBodies[index].body
                               == helperPulseBodies[index].body
            && sameVec(scalarPulseBodies[index].position,
                       helperPulseBodies[index].position)
            && sameVec(scalarPulseBodies[index].impulse,
                       helperPulseBodies[index].impulse)
            && scalarPulseBodies[index].surfaceDistance
                == helperPulseBodies[index].surfaceDistance;
    }
    scalarPulse.step(kFixedDt);
    helperPulse.step(kFixedDt);
    bool exactPulseStep = true;
    for (std::size_t slot = 0;
         exactPulseStep && slot < scalarPulseIds.size(); ++slot) {
        const RigidBody& scalarBody = scalarPulse.bodies()[
            static_cast<std::size_t>(scalarPulseIds[slot])];
        const RigidBody& helperBody = helperPulse.bodies()[
            static_cast<std::size_t>(helperPulseIds[slot])];
        exactPulseStep = sameVec(scalarBody.position, helperBody.position)
                      && sameVec(scalarBody.velocity, helperBody.velocity);
    }
    verify("echo pulse is deterministic across helper backends",
           scalarPulseEmitted && helperPulseEmitted
        && scalarPulseEvent && helperPulseEvent
        && scalarPulseEvent->serial == helperPulseEvent->serial
        && sameVec(scalarPulseEvent->source, helperPulseEvent->source)
        && sameVec(scalarPulseEvent->origin, helperPulseEvent->origin)
        && sameVec(scalarPulseEvent->direction, helperPulseEvent->direction)
        && scalarPulseEvent->simulationTime == helperPulseEvent->simulationTime
        && scalarPulseEvent->targetBody == helperPulseEvent->targetBody
        && scalarPulseEvent->affectedBodyCount
             == helperPulseEvent->affectedBodyCount
        && scalarPulseEvent->totalDeliveredImpulse
             == helperPulseEvent->totalDeliveredImpulse
        && exactPulseEvents && exactPulseStep);                                 // 143

    PhysicsWorld cappedPulse(7'305, 0);
    sleepAllBodies(cappedPulse);
    for (RigidBody& body : cappedPulse.bodies()) {
        body.position = {90.0, 9.0, 90.0};
        body.previousPosition = body.position;
    }
    cappedPulse.player().position = {40.0, 3.24, 32.0};
    cappedPulse.player().previousPosition = cappedPulse.player().position;
    cappedPulse.player().yaw = 0.0;
    cappedPulse.player().pitch = 0.0;
    constexpr int cappedTarget = 100;
    cappedPulse.bodies()[cappedTarget].position = {40.0, 5.0, 37.0};
    cappedPulse.bodies()[cappedTarget].previousPosition =
        cappedPulse.bodies()[cappedTarget].position;
    for (int index = 101; index <= 164; ++index) {
        RigidBody& body = cappedPulse.bodies()[static_cast<std::size_t>(index)];
        body.position = {40.4 + static_cast<Real>(index - 101) * 0.005,
                         5.0, 37.0};
        body.previousPosition = body.position;
    }
    const bool emittedCappedPulse = cappedPulse.emitEchoPulse();
    const auto cappedEvent = cappedPulse.lastEchoPulseEvent();
    const auto cappedBodies = cappedPulse.lastEchoPulseBodies();
    bool cappedOrder = true;
    bool cappedTargetRetained = false;
    for (std::size_t index = 0; index < cappedBodies.size(); ++index) {
        cappedTargetRetained = cappedTargetRetained
            || cappedBodies[index].body == cappedTarget;
        if (index == 0U) continue;
        const EchoPulseBodyEvent& previous = cappedBodies[index - 1U];
        const EchoPulseBodyEvent& current = cappedBodies[index];
        cappedOrder = cappedOrder
            && (previous.surfaceDistance < current.surfaceDistance
                || (previous.surfaceDistance == current.surfaceDistance
                    && previous.body < current.body));
    }
    verify("echo pulse cap retains target in deterministic surface order",
           emittedCappedPulse && cappedEvent
        && cappedEvent->targetBody == cappedTarget
        && cappedEvent->affectedBodyCount == kEchoPulseMaximumAffectedBodies
        && cappedBodies.size() == kEchoPulseMaximumAffectedBodies
        && cappedTargetRetained && cappedOrder);                                // 144

    if (verbose) {
        std::cout << kTitle << " deterministic self-check\n";
        for (const auto& label : passed) std::cout << "  PASS  " << label << '\n';
        for (const auto& label : failed) std::cout << "  FAIL  " << label << '\n';
        std::cout << "Result: " << passed.size() << " passed, "
                  << failed.size() << " failed\n";
    }
    return failed.empty();
}

} // namespace nec
