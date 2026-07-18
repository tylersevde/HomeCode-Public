#pragma once

#include "core.hpp"

#include <deque>

namespace nec {

enum class GalileoExperimentState : std::uint8_t {
    Inactive,
    Running,
    Complete,
};

struct GalileoBodyIds {
    int steel{-1};
    int concrete{-1};
};

// One paired, fixed-step sample keeps both trajectories on exactly the same
// timeline.  Renderers may draw these positions, but the authoritative run
// remains entirely owned by PhysicsWorld.
struct GalileoTrailSample {
    double elapsed{};
    Vec3 steel{};
    Vec3 concrete{};
};

enum class GravityInterceptState : std::uint8_t {
    Inactive,
    Running,
    Hit,
    Missed,
};

struct GravityInterceptBodyIds {
    int projectile{-1};
    int target{-1};
};

// A paired fixed-step history makes the falling target and projectile easy to
// compare under different gravity settings without moving any authority out
// of PhysicsWorld. The hit point itself is taken from the solver contact.
struct GravityInterceptTrailSample {
    double elapsed{};
    Vec3 projectile{};
    Vec3 target{};
};

// One authoritative pulse snapshot is retained until the next pulse or reset.
// Consumers can key transient graphics from serial without manufacturing a
// physics contact, while the per-body records expose only impulses that were
// actually delivered to the solver state.
struct EchoPulseBodyEvent {
    std::uint16_t body{};
    Vec3 position{};
    Vec3 impulse{};
    Real surfaceDistance{};
};

struct EchoPulseEvent {
    std::uint64_t serial{};
    Vec3 source{};
    Vec3 origin{};
    Vec3 direction{};
    double simulationTime{};
    Real radius{};
    int targetBody{-1};
    std::uint32_t affectedBodyCount{};
    Real totalDeliveredImpulse{};
};

class PhysicsWorld {
public:
    explicit PhysicsWorld(std::uint32_t seed = 1337, unsigned helpers = 3,
                          std::string_view cpuBackend = "auto");

    void reset();
    // A rendered frame can contain several fixed simulation steps. Accumulate
    // a bounded batch of their impact events for audiovisual consumers until
    // endFrameEvents(), while preserving the usual per-step event semantics
    // for standalone physics callers.
    void beginFrameEvents() noexcept;
    void endFrameEvents() noexcept;
    void step(Real dt = kFixedDt);
    void setMoveInput(Real forward, Real strafe) noexcept;
    void adjustGravity(int direction);
    void adjustFriction(int direction);
    void adjustThrowForce(int direction);
    void togglePlayerPositionLock();
    void startGalileoExperiment();
    void startGravityInterceptExperiment();
    [[nodiscard]] bool emitEchoPulse();
    [[nodiscard]] bool jump();
    [[nodiscard]] int raycastBody(const Vec3& origin, const Vec3& direction,
                                  Real reach = kPickupReach) const;
    [[nodiscard]] bool pickupOrDrop();
    [[nodiscard]] bool throwHeld();
    void wakeAll();
    void wakeNearby(const Vec3& position, Real radius = 0.34, int limit = 16);

    [[nodiscard]] const std::vector<BodySpec>& specs() const noexcept { return specs_; }
    [[nodiscard]] const std::vector<RigidBody>& bodies() const noexcept { return bodies_; }
    [[nodiscard]] std::vector<RigidBody>& bodies() noexcept { return bodies_; }
    [[nodiscard]] const Player& player() const noexcept { return player_; }
    [[nodiscard]] Player& player() noexcept { return player_; }
    [[nodiscard]] std::span<const ImpactEvent> impacts() const noexcept { return impacts_; }
    [[nodiscard]] std::string_view lastMessage() const noexcept;
    [[nodiscard]] Real gravity() const noexcept { return gravity_; }
    [[nodiscard]] Real roomFriction() const noexcept { return roomFriction_; }
    [[nodiscard]] Real throwForce() const noexcept { return throwForce_; }
    [[nodiscard]] bool playerPositionLocked() const noexcept {
        return playerPositionLocked_;
    }
    [[nodiscard]] double simulationTime() const noexcept { return simulationTime_; }
    [[nodiscard]] int heldBody() const noexcept { return heldBody_; }
    [[nodiscard]] std::size_t broadphaseCandidates() const noexcept {
        return lastBroadphaseCandidates_;
    }
    [[nodiscard]] std::size_t activeContacts() const noexcept { return lastActiveContacts_; }
    [[nodiscard]] int solverIterations() const noexcept { return lastSolverIterations_; }
    [[nodiscard]] double broadphaseMilliseconds() const noexcept {
        return broadphaseMilliseconds_;
    }
    [[nodiscard]] unsigned helperCount() const noexcept { return executor_.helperCount(); }
    [[nodiscard]] std::string_view cpuBackend() const noexcept { return cpuBackend_; }
    [[nodiscard]] GalileoExperimentState galileoState() const noexcept {
        return galileoState_;
    }
    [[nodiscard]] bool galileoRunning() const noexcept {
        return galileoState_ == GalileoExperimentState::Running;
    }
    [[nodiscard]] std::string_view galileoStatus() const noexcept;
    [[nodiscard]] GalileoBodyIds galileoBodyIds() const noexcept {
        return galileoBodyIds_;
    }
    [[nodiscard]] double galileoElapsed() const noexcept {
        return galileoElapsed_;
    }
    [[nodiscard]] std::optional<double> galileoSteelLandingTime() const noexcept {
        return galileoSteelLandingTime_;
    }
    [[nodiscard]] std::optional<double> galileoConcreteLandingTime() const noexcept {
        return galileoConcreteLandingTime_;
    }
    [[nodiscard]] std::span<const GalileoTrailSample>
    galileoCurrentTrail() const noexcept {
        return galileoCurrentTrail_;
    }
    [[nodiscard]] std::span<const GalileoTrailSample>
    galileoPreviousTrail() const noexcept {
        return galileoPreviousTrail_;
    }
    [[nodiscard]] GravityInterceptState gravityInterceptState() const noexcept {
        return gravityInterceptState_;
    }
    [[nodiscard]] bool gravityInterceptRunning() const noexcept {
        return gravityInterceptState_ == GravityInterceptState::Running;
    }
    [[nodiscard]] std::string_view gravityInterceptStatus() const noexcept;
    [[nodiscard]] GravityInterceptBodyIds
    gravityInterceptBodyIds() const noexcept {
        return gravityInterceptBodyIds_;
    }
    [[nodiscard]] double gravityInterceptElapsed() const noexcept {
        return gravityInterceptElapsed_;
    }
    [[nodiscard]] std::optional<double>
    gravityInterceptHitTime() const noexcept {
        return gravityInterceptHitTime_;
    }
    [[nodiscard]] std::optional<Vec3>
    gravityInterceptHitPoint() const noexcept {
        return gravityInterceptHitPoint_;
    }
    [[nodiscard]] std::optional<Real>
    gravityInterceptClosestGap() const noexcept {
        return gravityInterceptClosestGap_;
    }
    [[nodiscard]] std::span<const GravityInterceptTrailSample>
    gravityInterceptCurrentTrail() const noexcept {
        return gravityInterceptCurrentTrail_;
    }
    [[nodiscard]] std::span<const GravityInterceptTrailSample>
    gravityInterceptPreviousTrail() const noexcept {
        return gravityInterceptPreviousTrail_;
    }
    [[nodiscard]] std::optional<EchoPulseEvent>
    lastEchoPulseEvent() const noexcept {
        return lastEchoPulseEvent_;
    }
    [[nodiscard]] std::span<const EchoPulseBodyEvent>
    lastEchoPulseBodies() const noexcept {
        return lastEchoPulseBodies_;
    }
    [[nodiscard]] double echoPulseCooldownRemaining() const noexcept;
    [[nodiscard]] bool echoPulseReady() const noexcept {
        return echoPulseCooldownRemaining() <= 0.0;
    }
    [[nodiscard]] std::string_view echoPulseStatus() const noexcept;

    [[nodiscard]] Real dynamicMass(const RigidBody& body) const noexcept;
    [[nodiscard]] Real inverseMass(const RigidBody& body) const noexcept;
    [[nodiscard]] Vec3 supportPoint(const RigidBody& body, const Vec3& direction) const noexcept;
    [[nodiscard]] Real supportExtent(const RigidBody& body, const Vec3& axis) const noexcept;

private:
    struct Contact {
        std::uint16_t first{};
        std::uint16_t second{};
        Vec3 normal{};
        Vec3 point{};
        Real penetration{};
        Real toi{1.0};
    };

    struct PendingWake {
        std::uint16_t body{};
        Vec3 impulse{};
        Vec3 point{};
    };

    struct alignas(64) PairBuffer {
        std::vector<Pair> pairs;
    };

    void spawn(std::uint16_t spec, const Vec3& position, std::string label = {},
               int groupIndex = 0, bool asleep = false, bool grounded = false,
               Quat orientation = {}, Vec3 color = {-1.0F, -1.0F, -1.0F});
    [[nodiscard]] std::uint16_t specIndex(std::string_view key) const;
    void wake(RigidBody& body, bool dirty = true) noexcept;
    void integratePlayer(Real dt);
    void holdConstraint(RigidBody& body);
    void integrateBody(std::size_t index, Real dt);
    void advanceBodySwept(RigidBody& body, Real dt);
    void projectBodyInsideRoom(RigidBody& body) const;
    void projectPlayerInsideRoom() noexcept;
    [[nodiscard]] Vec3 bodyDrag(const RigidBody& body) const noexcept;
    [[nodiscard]] Real effectiveRoomFriction(const RigidBody& body) const noexcept;

    void buildBroadphasePairs(std::span<const std::uint16_t> active);
    void buildSpatialBroadphasePairs(std::span<const std::uint16_t> active);
    [[nodiscard]] bool sweptBoundingSpheres(const RigidBody& a,
                                            const RigidBody& b) const noexcept;
    [[nodiscard]] std::optional<Contact> findContact(std::uint16_t first,
                                                     std::uint16_t second) const;
    [[nodiscard]] std::optional<Contact> sphereSphereContact(
        std::uint16_t first, std::uint16_t second) const;
    [[nodiscard]] std::optional<Contact> sphereBoxContact(
        std::uint16_t sphere, std::uint16_t box, bool flip) const;
    [[nodiscard]] std::optional<Contact> boxBoxContact(
        std::uint16_t first, std::uint16_t second) const;
    [[nodiscard]] Real applyBodyImpulse(const Contact& contact,
                                        bool positionCorrection);
    void resolvePlayerBody(std::uint16_t body, Real dt);
    void groundResistanceAndSleep(RigidBody& body, Real dt);
    void updateAttachments();
    void updateWheelbarrowCargo(Real dt);
    void drainPendingWakes();
    void enforcePlayerPositionLock() noexcept;
    [[nodiscard]] bool isGalileoBody(std::size_t index) const noexcept;
    [[nodiscard]] bool isGravityInterceptBody(std::size_t index) const noexcept;
    [[nodiscard]] bool isIdealVacuumBody(std::size_t index) const noexcept;
    void updateGalileoExperiment(Real dt);
    void updateGravityInterceptExperiment(Real dt);
    void releaseHeldBodyForExperiment() noexcept;
    void detachPayloadsFromExperimentBodies(std::span<const int> ids);
    void prepareExperimentBody(int id, const Vec3& position,
                               const Vec3& velocity = {});
    void addMessage(std::string message);

    std::uint32_t seed_{};
    std::mt19937 rng_;
    std::vector<BodySpec> specs_;
    std::unordered_map<std::string, std::uint16_t> specByKey_;
    std::vector<RigidBody> bodies_;
    Player player_{};
    Vec3 lockedPlayerPosition_{player_.position};
    bool playerPositionLocked_{};
    ParallelExecutor executor_;
    std::vector<Pair> broadphasePairs_;
    std::vector<PairBuffer> threadPairs_;
    std::vector<Contact> contacts_;
    std::vector<ImpactEvent> impacts_;
    bool collectingFrameEvents_{};
    std::vector<PendingWake> pendingWakes_;
    std::deque<std::string> messages_;
    std::vector<std::uint16_t> activeIndices_;
    int heldBody_{-1};
    int wheelbarrowIndex_{-1};
    Real gravity_{kEarthGravity};
    Real roomFriction_{kDefaultRoomFriction};
    Real throwForce_{kThrowForceMin};
    double simulationTime_{};
    std::size_t lastBroadphaseCandidates_{};
    std::size_t lastActiveContacts_{};
    int lastSolverIterations_{kSolverIterations};
    double broadphaseMilliseconds_{};
    std::string cpuBackend_{"auto"};
    GalileoExperimentState galileoState_{GalileoExperimentState::Inactive};
    GalileoBodyIds galileoBodyIds_{};
    double galileoElapsed_{};
    std::optional<double> galileoSteelLandingTime_;
    std::optional<double> galileoConcreteLandingTime_;
    std::vector<GalileoTrailSample> galileoCurrentTrail_;
    std::vector<GalileoTrailSample> galileoPreviousTrail_;
    GravityInterceptState gravityInterceptState_{
        GravityInterceptState::Inactive};
    GravityInterceptBodyIds gravityInterceptBodyIds_{};
    double gravityInterceptElapsed_{};
    std::optional<double> gravityInterceptHitTime_;
    std::optional<Vec3> gravityInterceptHitPoint_;
    std::optional<Real> gravityInterceptClosestGap_;
    std::vector<GravityInterceptTrailSample> gravityInterceptCurrentTrail_;
    std::vector<GravityInterceptTrailSample> gravityInterceptPreviousTrail_;
    std::optional<EchoPulseEvent> lastEchoPulseEvent_;
    std::vector<EchoPulseBodyEvent> lastEchoPulseBodies_;
    std::uint64_t nextEchoPulseSerial_{1U};
};

} // namespace nec
