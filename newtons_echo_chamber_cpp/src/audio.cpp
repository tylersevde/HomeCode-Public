#include "audio.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif

namespace nec {
namespace {

constexpr int kSampleRate = 44'100;
constexpr std::size_t kBlockFrames = 256;
constexpr float kPi = std::numbers::pi_v<float>;
constexpr float kRoomSignalLimit = 32'767.0F;
constexpr float kRoomStateLimit = 49'152.0F;
constexpr float kRoomWetLimit = 18'000.0F;
#if defined(__linux__)
constexpr std::size_t kWriteChunkFrames = 128;
constexpr auto kWritePollInterval = std::chrono::milliseconds{12};
constexpr auto kChildExitGrace = std::chrono::milliseconds{180};
constexpr auto kTransportStartupProbe = std::chrono::milliseconds{72};
constexpr auto kTransportProbeInterval = std::chrono::milliseconds{4};
constexpr char kLowLatencyBufferTime[] = "80000";
constexpr char kLowLatencyPeriodTime[] = "20000";
#endif

float envelope(float t, float rate) noexcept { return std::exp(-rate * t); }

struct ReflectionTap {
    std::size_t delayFrames;
    float leftGain;
    float rightGain;
    float damping;
};

// Six image-source-style arrivals for the 100 x 100 x 10 metre chamber. The
// deliberately unequal paths keep repeated impacts from sounding like one
// metallic slap, and the per-wall panning makes the room audible in stereo.
constexpr std::array<ReflectionTap, 6> kReflectionTaps{{
    {397U, 0.135F, 0.115F, 0.22F},  // floor
    {683U, 0.170F, 0.045F, 0.30F},  // left wall
    {911U, 0.045F, 0.165F, 0.31F},  // right wall
    {1'277U, 0.115F, 0.105F, 0.38F}, // front wall
    {1'901U, 0.085F, 0.100F, 0.45F}, // back wall
    {2'699U, 0.065F, 0.080F, 0.52F}, // ceiling
}};

// Mutually prime-ish lengths prevent the four-line feedback network from
// collapsing into a short, pitched repeat. All fit in the fixed 2048-frame
// storage declared by AudioEngine.
constexpr std::array<std::size_t, 4> kReverbDelayFrames{
    1'559U, 1'663U, 1'789U, 1'907U,
};
constexpr std::array<float, 4> kReverbInputSigns{
    0.5F, -0.5F, 0.5F, 0.5F,
};
constexpr float kReverbFeedback = 0.74F;
constexpr float kReverbDampingBlend = 0.34F;
constexpr float kReverbOutputGain = 0.36F;

float suppressDenormal(float value) noexcept {
    return std::abs(value) < 1.0e-12F ? 0.0F : value;
}

float roomSendFor(std::string_view key) noexcept {
    // Hard, resonant objects illuminate the room more than absorptive props.
    if (key == "echo_pulse") return 0.58F;
    if (key == "intercept") return 0.48F;
    if (key == "ceramic") return 0.42F;
    if (key == "steel") return 0.39F;
    if (key == "wheelbarrow" || key == "bowling") return 0.35F;
    if (key == "concrete") return 0.31F;
    if (key == "clay_brick") return 0.29F;
    if (key == "wood_bat" || key == "timber" || key == "pallet") return 0.27F;
    if (key == "land") return 0.23F;
    if (key == "rubber_brick" || key == "balloon" || key == "throw") {
        return 0.19F;
    }
    if (key == "foam_noodle" || key == "goo" || key == "plush") return 0.11F;
    if (key == "medicine") return 0.08F;
    return 0.16F;
}

} // namespace

AudioEngine::AudioEngine(bool enabled) {
    if (!enabled) return;
    buildBank();
    if (startTransport()) {
        thread_ = std::jthread([this](std::stop_token stop) { audioLoop(stop); });
    }
}

AudioEngine::~AudioEngine() {
    if (thread_.joinable()) {
        thread_.request_stop();
        // The transport is nonblocking and poll uses a short finite timeout,
        // so the mixer always observes this request promptly.
        thread_.join();
    }
    stopTransport();
}

void AudioEngine::toggleMute() noexcept {
    bool wasMuted = muted_.load(std::memory_order_relaxed);
    while (!muted_.compare_exchange_weak(
        wasMuted, !wasMuted, std::memory_order_acq_rel,
        std::memory_order_relaxed)) {
    }
    if (!wasMuted) {
        // The audio thread observes this generation and exclusively clears
        // all live voices and room state. Keeping that state single-threaded
        // avoids a data race with the mixer while the atomic flag gates the
        // very next sample written by this process.
        (void)muteGeneration_.fetch_add(1U, std::memory_order_release);
    }
}

bool AudioEngine::startTransport() noexcept {
#if defined(__linux__)
    struct SpawnedTransport {
        int writeFd{-1};
        pid_t child{-1};
    };

    const auto spawnAttempt = [](bool lowLatency) noexcept {
        SpawnedTransport result{};
        int descriptors[2]{-1, -1};
        if (::pipe2(descriptors, O_CLOEXEC) != 0) return result;

        const int readFd = descriptors[0];
        const int writeFd = descriptors[1];
        const int currentFlags = ::fcntl(writeFd, F_GETFL, 0);
        if (currentFlags < 0 ||
            ::fcntl(writeFd, F_SETFL, currentFlags | O_NONBLOCK) != 0) {
            ::close(readFd);
            ::close(writeFd);
            return result;
        }
#if defined(F_SETPIPE_SZ)
        // Do not let the transport pipe silently become a second, hundreds-of-
        // milliseconds audio buffer. Linux may round this up to one page.
        (void)::fcntl(writeFd, F_SETPIPE_SZ, 4'096);
#endif

        const int nullFd = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
        posix_spawn_file_actions_t actions{};
        if (::posix_spawn_file_actions_init(&actions) != 0) {
            if (nullFd >= 0) ::close(nullFd);
            ::close(readFd);
            ::close(writeFd);
            return result;
        }

        int actionError = 0;
        if (readFd != STDIN_FILENO) {
            actionError = ::posix_spawn_file_actions_adddup2(
                &actions, readFd, STDIN_FILENO);
        }
        if (actionError == 0 && readFd != STDIN_FILENO) {
            actionError = ::posix_spawn_file_actions_addclose(&actions, readFd);
        }
        if (actionError == 0 && writeFd != STDIN_FILENO) {
            actionError = ::posix_spawn_file_actions_addclose(&actions, writeFd);
        }
        if (actionError == 0 && nullFd >= 0 && nullFd != STDERR_FILENO) {
            actionError = ::posix_spawn_file_actions_adddup2(
                &actions, nullFd, STDERR_FILENO);
        }
        if (actionError == 0 && nullFd >= 0 && nullFd != STDERR_FILENO) {
            actionError = ::posix_spawn_file_actions_addclose(
                &actions, nullFd);
        }

        pid_t child = -1;
        if (actionError == 0) {
            std::array<char*, 16> arguments{};
            std::size_t argument = 0U;
            arguments[argument++] = const_cast<char*>("aplay");
            arguments[argument++] = const_cast<char*>("-q");
            arguments[argument++] = const_cast<char*>("-t");
            arguments[argument++] = const_cast<char*>("raw");
            arguments[argument++] = const_cast<char*>("-f");
            arguments[argument++] = const_cast<char*>("S16_LE");
            arguments[argument++] = const_cast<char*>("-r");
            arguments[argument++] = const_cast<char*>("44100");
            arguments[argument++] = const_cast<char*>("-c");
            arguments[argument++] = const_cast<char*>("2");
            if (lowLatency) {
                arguments[argument++] = const_cast<char*>("-B");
                arguments[argument++] = const_cast<char*>(
                    kLowLatencyBufferTime);
                arguments[argument++] = const_cast<char*>("-F");
                arguments[argument++] = const_cast<char*>(
                    kLowLatencyPeriodTime);
            }
            arguments[argument] = nullptr;
            actionError = ::posix_spawnp(
                &child, "aplay", &actions, nullptr, arguments.data(),
                ::environ);
        }
        (void)::posix_spawn_file_actions_destroy(&actions);
        if (nullFd >= 0) ::close(nullFd);
        ::close(readFd);
        if (actionError != 0 || child <= 0) {
            ::close(writeFd);
            return result;
        }
        result.writeFd = writeFd;
        result.child = child;
        return result;
    };

    const auto survivesStartup = [](pid_t child) noexcept {
        const auto deadline = std::chrono::steady_clock::now()
            + kTransportStartupProbe;
        while (std::chrono::steady_clock::now() < deadline) {
            int status = 0;
            const pid_t waited = ::waitpid(child, &status, WNOHANG);
            if (waited == child) return false;
            if (waited < 0 && errno != EINTR) return false;
            std::this_thread::sleep_for(kTransportProbeInterval);
        }
        return true;
    };

    // Four periods in an 80 ms ALSA buffer is conservative enough for Pi-class
    // USB/HDMI devices while avoiding aplay's often much larger default. Some
    // ALSA plugins reject explicit timing; retry the original compatible form
    // whenever that process exits during parameter negotiation.
    SpawnedTransport transport = spawnAttempt(true);
    bool transportReady = transport.child > 0 &&
        survivesStartup(transport.child);
    if (!transportReady) {
        if (transport.writeFd >= 0) (void)::close(transport.writeFd);
        transport = spawnAttempt(false);
        transportReady = transport.child > 0 &&
            survivesStartup(transport.child);
    }
    if (!transportReady) {
        if (transport.writeFd >= 0) (void)::close(transport.writeFd);
        return false;
    }

    writeFd_ = transport.writeFd;
    childPid_ = static_cast<int>(transport.child);
    available_.store(true, std::memory_order_release);
    return true;
#else
    return false;
#endif
}

void AudioEngine::stopTransport() noexcept {
    available_.store(false, std::memory_order_release);
#if defined(__linux__)
    if (writeFd_ >= 0) {
        // EOF gives a healthy aplay process a clean, immediate exit path.
        (void)::close(writeFd_);
        writeFd_ = -1;
    }
    if (childPid_ <= 0) return;

    const pid_t child = static_cast<pid_t>(childPid_);
    const auto waitForExit = [child](std::chrono::milliseconds budget) noexcept {
        const auto deadline = std::chrono::steady_clock::now() + budget;
        for (;;) {
            int status = 0;
            const pid_t result = ::waitpid(child, &status, WNOHANG);
            if (result == child) return true;
            if (result < 0 && errno == ECHILD) return true;
            if (result < 0 && errno != EINTR) return true;
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds{6});
        }
    };

    if (!waitForExit(kChildExitGrace)) {
        (void)::kill(child, SIGTERM);
        if (!waitForExit(kChildExitGrace)) {
            (void)::kill(child, SIGKILL);
            // Never turn shutdown back into an unbounded wait, even if a
            // broken device leaves the child in an uninterruptible kernel
            // state. A normal SIGKILL path is still reaped here.
            (void)waitForExit(kChildExitGrace);
        }
    }
    childPid_ = -1;
#else
    writeFd_ = -1;
    childPid_ = -1;
#endif
}

bool AudioEngine::writeAudioBlock(std::stop_token stop,
                                  const std::int16_t* samples,
                                  std::size_t sampleCount) noexcept {
#if defined(__linux__)
    if (writeFd_ < 0 || samples == nullptr) return false;
    const auto* cursor = reinterpret_cast<const std::uint8_t*>(samples);
    std::size_t remaining = sampleCount * sizeof(std::int16_t);
    constexpr std::size_t kWriteChunkBytes =
        kWriteChunkFrames * 2U * sizeof(std::int16_t);
    const std::array<std::uint8_t, kWriteChunkBytes> silence{};
    while (remaining > 0 && !stop.stop_requested()) {
        const std::size_t chunk = std::min(remaining, kWriteChunkBytes);
        const bool silenceOutput = muted_.load(std::memory_order_acquire);
        const std::uint8_t* writeCursor =
            silenceOutput ? silence.data() : cursor;
        const ssize_t written = ::write(writeFd_, writeCursor, chunk);
        if (written > 0) {
            const auto consumed = static_cast<std::size_t>(written);
            cursor += consumed;
            remaining -= consumed;
            continue;
        }
        if (written < 0 && errno == EINTR) continue;
        if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd descriptor{writeFd_, POLLOUT, 0};
            const int ready = ::poll(&descriptor, 1,
                                     static_cast<int>(kWritePollInterval.count()));
            if (ready < 0 && errno == EINTR) continue;
            if (ready < 0) return false;
            if (ready > 0 &&
                (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                return false;
            }
            continue;
        }
        return false;
    }
    return remaining == 0;
#else
    (void)stop;
    (void)samples;
    (void)sampleCount;
    return false;
#endif
}

std::vector<std::int16_t> AudioEngine::synthesize(
    float seconds, const std::function<float(float, std::size_t)>& sample) {
    const auto frames = static_cast<std::size_t>(
        std::max(1.0F, seconds * static_cast<float>(kSampleRate)));
    std::vector<std::int16_t> output(frames * 2U);
    for (std::size_t i = 0; i < frames; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kSampleRate);
        const float value = clamp(sample(t, i), -1.0F, 1.0F);
        const auto shaped = static_cast<std::int16_t>(value * 27'500.0F);
        output[i * 2U] = shaped;
        output[i * 2U + 1U] = shaped;
    }
    return output;
}

void AudioEngine::buildBank() {
    auto tone = [](float base, float decay, float second, float seconds) {
        return synthesize(seconds, [=](float t, std::size_t) {
            return envelope(t, decay) *
                (0.72F * std::sin(2.0F * kPi * base * t) +
                 0.23F * std::sin(2.0F * kPi * base * second * t));
        });
    };
    bank_["dodge"] = tone(138.0F, 8.2F, 2.03F, 0.34F);
    bank_["medicine"] = tone(47.0F, 18.0F, 1.31F, 0.32F);
    bank_["steel"] = synthesize(0.62F, [](float t, std::size_t) {
        return std::min(1.0F, t * 380.0F) * envelope(t, 4.9F) *
            (0.46F * std::sin(2.0F * kPi * 713.0F * t) +
             0.28F * std::sin(2.0F * kPi * 1147.0F * t + 0.4F) +
             0.18F * std::sin(2.0F * kPi * 1933.0F * t + 1.1F));
    });
    bank_["concrete"] = tone(83.0F, 25.0F, 2.61F, 0.30F);
    bank_["rubber_brick"] = tone(92.0F, 12.0F, 1.70F, 0.31F);
    bank_["wood_bat"] = tone(244.0F, 10.0F, 2.42F, 0.40F);
    bank_["wheelbarrow"] = tone(119.0F, 7.5F, 4.33F, 0.46F);
    bank_["balloon"] = tone(312.0F, 16.0F, 1.04F, 0.30F);
    bank_["foam_noodle"] = tone(176.0F, 14.0F, 1.51F, 0.28F);
    bank_["goo"] = tone(61.0F, 15.0F, 1.12F, 0.34F);
    bank_["ceramic"] = tone(1280.0F, 8.0F, 1.67F, 0.45F);
    bank_["bowling"] = synthesize(0.52F, [](float t, std::size_t sampleIndex) {
        const float laneKnock = envelope(t, 11.0F)
            * (0.55F * std::sin(2.0F * kPi * 112.0F * t)
               + 0.19F * std::sin(2.0F * kPi * 347.0F * t + 0.22F));
        const float resinClick = envelope(t, 28.0F)
            * (0.16F * std::sin(2.0F * kPi * 921.0F * t)
               + 0.07F * std::sin(static_cast<float>(sampleIndex) * 1.731F));
        return laneKnock + resinClick;
    });
    bank_["timber"] = synthesize(0.58F, [](float t, std::size_t sampleIndex) {
        const float hollowThud = envelope(t, 8.5F)
            * (0.58F * std::sin(2.0F * kPi * 76.0F * t)
               + 0.21F * std::sin(2.0F * kPi * 163.0F * t + 0.35F)
               + 0.10F * std::sin(2.0F * kPi * 271.0F * t + 0.91F));
        const float barkCrack = envelope(t, 34.0F) * 0.08F
            * std::sin(static_cast<float>(sampleIndex) * 2.137F + 0.4F);
        return hollowThud + barkCrack;
    });
    bank_["clay_brick"] = tone(102.0F, 22.0F, 2.15F, 0.34F);
    bank_["pallet"] = tone(72.0F, 13.0F, 2.83F, 0.42F);
    bank_["plush"] = tone(164.0F, 20.0F, 1.07F, 0.25F);
    bank_["throw"] = synthesize(0.24F, [](float t, std::size_t) {
        const float sweep = 105.0F + 260.0F * t;
        return envelope(t, 11.0F) * std::sin(2.0F * kPi * sweep * t);
    });
    bank_["jump"] = synthesize(0.22F, [](float t, std::size_t) {
        return envelope(t, 10.0F) *
            (0.65F * std::sin(2.0F * kPi * (120.0F + 190.0F * t) * t) +
             0.18F * std::sin(2.0F * kPi * 57.0F * t));
    });
    bank_["land"] = tone(54.0F, 24.0F, 1.92F, 0.28F);
    bank_["intercept"] = synthesize(0.72F, [](float t, std::size_t sampleIndex) {
        const float attack = std::min(1.0F, t * 42.0F);
        const float chirp = 118.0F + 520.0F * t * t;
        const float pulse = std::sin(2.0F * kPi * chirp * t)
            * (0.52F + 0.18F * std::sin(2.0F * kPi * 7.0F * t));
        const float clock = 0.12F * std::sin(
            static_cast<float>(sampleIndex) * 0.071F + t * 9.0F);
        return attack * envelope(t, 3.9F) * (pulse + clock);
    });
    bank_["echo_pulse"] = synthesize(
        0.86F, [](float t, std::size_t sampleIndex) {
            const float attack = std::min(1.0F, t * 90.0F);
            const float expandingPhase = 2.0F * kPi
                * (92.0F * t + 215.0F * t * t - 54.0F * t * t * t);
            const float sonar = 0.54F * std::sin(expandingPhase)
                              + 0.20F * std::sin(expandingPhase * 2.013F + 0.3F);
            const float sub = 0.24F * std::sin(
                2.0F * kPi * (38.0F + 18.0F * t) * t);
            const float deterministicAir = 0.07F * std::sin(
                static_cast<float>(sampleIndex) * 1.719F + t * 13.0F);
            return attack * envelope(t, 3.6F)
                 * (sonar + sub + deterministicAir);
        });
}

bool AudioEngine::play(std::string_view key, float volume, float pan) {
    if (!available_.load(std::memory_order_acquire) ||
        muted_.load(std::memory_order_acquire)) return false;
    const std::uint64_t muteGeneration =
        muteGeneration_.load(std::memory_order_acquire);
    const auto found = bank_.find(std::string(key));
    if (found == bank_.end()) return false;
    const float safePan = clamp(pan, -1.0F, 1.0F);
    const float safeVolume = clamp(volume, 0.0F, 1.0F);
    std::lock_guard lock(queueMutex_);
    if (muted_.load(std::memory_order_acquire) ||
        muteGeneration !=
            muteGeneration_.load(std::memory_order_acquire)) {
        return false;
    }
    if (requestCount_ == kRequestCapacity) {
        requestHead_ = (requestHead_ + 1U) % kRequestCapacity;
        --requestCount_;
    }
    const std::size_t tail = (requestHead_ + requestCount_) % kRequestCapacity;
    requests_[tail] = {
        &found->second,
        safeVolume * (1.0F - safePan) * 0.5F,
        safeVolume * (1.0F + safePan) * 0.5F,
        roomSendFor(key),
        muteGeneration,
    };
    ++requestCount_;
    return true;
}

bool AudioEngine::playImpact(const ImpactEvent& event,
                             const Player& listener) {
    const Vec3 offset = event.position - listener.eye();
    const Real distance = offset.length();
    if (distance > 45.0F) return false;
    const Real pan = clamp(offset.dot(listener.right()) / std::max(3.0, distance),
                           -0.85, 0.85);
    const Real attenuation = 1.0 / (1.0 + 0.055 * distance * distance);
    const Real strength = clamp(std::log1p(event.impulse) / 8.0, 0.08, 1.0);
    return play(event.family, static_cast<float>(attenuation * strength),
                static_cast<float>(pan));
}

AudioEngine::StereoFrame AudioEngine::processRoom(float input) noexcept {
    static_assert(kReflectionTaps.size() == kWallCount);
    static_assert(kReverbDelayFrames.size() == kReverbLineCount);
    static_assert((kEarlyDelayCapacity & (kEarlyDelayCapacity - 1U)) == 0U);
    static_assert(2'699U < kEarlyDelayCapacity);
    static_assert(1'907U < kReverbDelayCapacity);

    const float finiteInput = std::isfinite(input) ? input : 0.0F;
    const float boundedInput = clamp(finiteInput, -kRoomSignalLimit,
                                     kRoomSignalLimit);

    // Remove any DC bias before it can circulate through the feedback network.
    float roomInput = boundedInput - roomDcInput_ + 0.995F * roomDcOutput_;
    roomDcInput_ = boundedInput;
    roomDcOutput_ = suppressDenormal(roomInput);
    roomInput = roomDcOutput_;

    earlyDelay_[earlyCursor_] = roomInput;
    float earlyLeft = 0.0F;
    float earlyRight = 0.0F;
    for (std::size_t wall = 0; wall < kReflectionTaps.size(); ++wall) {
        const ReflectionTap& tap = kReflectionTaps[wall];
        const std::size_t read =
            (earlyCursor_ + kEarlyDelayCapacity - tap.delayFrames)
            & (kEarlyDelayCapacity - 1U);
        const float delayed = earlyDelay_[read];
        float filtered = earlyLowPass_[wall]
            + (delayed - earlyLowPass_[wall]) * (1.0F - tap.damping);
        filtered = suppressDenormal(filtered);
        earlyLowPass_[wall] = filtered;
        earlyLeft += filtered * tap.leftGain;
        earlyRight += filtered * tap.rightGain;
    }
    earlyCursor_ = (earlyCursor_ + 1U) & (kEarlyDelayCapacity - 1U);

    std::array<float, kReverbLineCount> delayed{};
    for (std::size_t line = 0; line < kReverbLineCount; ++line) {
        delayed[line] =
            reverbDelay_[line][reverbCursor_[line]];
    }

    // An orthonormal 4x4 Hadamard mix preserves energy; feedback below unity
    // therefore remains stable even before damping and the explicit bounds.
    const std::array<float, kReverbLineCount> mixed{
        0.5F * (delayed[0] + delayed[1] + delayed[2] + delayed[3]),
        0.5F * (delayed[0] - delayed[1] + delayed[2] - delayed[3]),
        0.5F * (delayed[0] + delayed[1] - delayed[2] - delayed[3]),
        0.5F * (delayed[0] - delayed[1] - delayed[2] + delayed[3]),
    };
    const float earlyMono = 0.5F * (earlyLeft + earlyRight);
    const float reverbInput = clamp(0.24F * roomInput + 0.72F * earlyMono,
                                    -kRoomSignalLimit, kRoomSignalLimit);
    for (std::size_t line = 0; line < kReverbLineCount; ++line) {
        float filtered = reverbLowPass_[line]
            + (mixed[line] - reverbLowPass_[line]) * kReverbDampingBlend;
        filtered = suppressDenormal(filtered);
        reverbLowPass_[line] = filtered;
        const float next = reverbInput * kReverbInputSigns[line]
            + kReverbFeedback * filtered;
        reverbDelay_[line][reverbCursor_[line]] =
            clamp(next, -kRoomStateLimit, kRoomStateLimit);
        ++reverbCursor_[line];
        if (reverbCursor_[line] == kReverbDelayFrames[line]) {
            reverbCursor_[line] = 0U;
        }
    }

    const float reverbLeft = 0.25F
        * (delayed[0] + delayed[1] - delayed[2] + delayed[3]);
    const float reverbRight = 0.25F
        * (delayed[0] - delayed[1] + delayed[2] + delayed[3]);
    return {
        clamp(earlyLeft + kReverbOutputGain * reverbLeft,
              -kRoomWetLimit, kRoomWetLimit),
        clamp(earlyRight + kReverbOutputGain * reverbRight,
              -kRoomWetLimit, kRoomWetLimit),
    };
}

void AudioEngine::resetMixerState() noexcept {
    voices_.fill({});
    voiceCount_ = 0U;
    earlyDelay_.fill(0.0F);
    earlyLowPass_.fill(0.0F);
    earlyCursor_ = 0U;
    for (auto& line : reverbDelay_) line.fill(0.0F);
    reverbCursor_.fill(0U);
    reverbLowPass_.fill(0.0F);
    roomDcInput_ = 0.0F;
    roomDcOutput_ = 0.0F;
}

void AudioEngine::audioLoop(std::stop_token stop) {
#if defined(__linux__)
    // aplay can disappear (or the selected ALSA device can fail) after the
    // transport starts. Block SIGPIPE on this writer thread so write() reports
    // EPIPE instead of terminating the entire game process.
    sigset_t blockedSignals;
    sigemptyset(&blockedSignals);
    sigaddset(&blockedSignals, SIGPIPE);
    (void)pthread_sigmask(SIG_BLOCK, &blockedSignals, nullptr);
#endif
    std::array<std::int16_t, kBlockFrames * 2U> output{};
    while (!stop.stop_requested() &&
           available_.load(std::memory_order_acquire)) {
        const std::uint64_t blockMuteGeneration =
            muteGeneration_.load(std::memory_order_acquire);
        if (blockMuteGeneration != mixerMuteGeneration_) {
            resetMixerState();
            mixerMuteGeneration_ = blockMuteGeneration;
        }
        {
            std::lock_guard lock(queueMutex_);
            while (requestCount_ > 0U) {
                const VoiceRequest request = requests_[requestHead_];
                requestHead_ = (requestHead_ + 1U) % kRequestCapacity;
                --requestCount_;
                if (request.samples != nullptr &&
                    request.muteGeneration == mixerMuteGeneration_) {
                    if (voiceCount_ == kVoiceCapacity) {
                        for (std::size_t voice = 1U;
                             voice < kVoiceCapacity; ++voice) {
                            voices_[voice - 1U] = voices_[voice];
                        }
                        --voiceCount_;
                    }
                    voices_[voiceCount_] = {
                        request.samples, 0U, request.left, request.right,
                        request.roomSend,
                    };
                    ++voiceCount_;
                }
            }
        }
        output.fill(0);
        for (std::size_t frame = 0; frame < kBlockFrames; ++frame) {
            // Check each generated frame, not just each 256-frame block. A mute
            // therefore cannot leave a direct voice or room tail audible until
            // the next callback. The comparatively large state reset happens
            // only once per mute transition.
            const std::uint64_t frameMuteGeneration =
                muteGeneration_.load(std::memory_order_acquire);
            if (frameMuteGeneration != mixerMuteGeneration_) {
                resetMixerState();
                mixerMuteGeneration_ = frameMuteGeneration;
            }
            float left = 0.0F;
            float right = 0.0F;
            float roomInput = 0.0F;
            for (std::size_t voiceIndex = 0U;
                 voiceIndex < voiceCount_; ++voiceIndex) {
                Voice& voice = voices_[voiceIndex];
                if (voice.samples == nullptr || voice.frame * 2U + 1U >= voice.samples->size()) {
                    continue;
                }
                const float sourceLeft = static_cast<float>(
                    (*voice.samples)[voice.frame * 2U]);
                const float sourceRight = static_cast<float>(
                    (*voice.samples)[voice.frame * 2U + 1U]);
                left += sourceLeft * voice.left;
                right += sourceRight * voice.right;
                roomInput += 0.5F * (sourceLeft + sourceRight)
                    * (voice.left + voice.right) * voice.roomSend;
                ++voice.frame;
            }
            const StereoFrame room = processRoom(roomInput);
            left += room.left;
            right += room.right;
            if (!muted_.load(std::memory_order_acquire)) {
                output[frame * 2U] = static_cast<std::int16_t>(
                    clamp(left, -32767.0F, 32767.0F));
                output[frame * 2U + 1U] = static_cast<std::int16_t>(
                    clamp(right, -32767.0F, 32767.0F));
            }
        }
        std::size_t liveVoices = 0U;
        for (std::size_t voiceIndex = 0U;
             voiceIndex < voiceCount_; ++voiceIndex) {
            const Voice& voice = voices_[voiceIndex];
            if (voice.samples == nullptr
                || voice.frame * 2U >= voice.samples->size()) {
                continue;
            }
            if (liveVoices != voiceIndex) voices_[liveVoices] = voice;
            ++liveVoices;
        }
        for (std::size_t voiceIndex = liveVoices;
             voiceIndex < voiceCount_; ++voiceIndex) {
            voices_[voiceIndex] = {};
        }
        voiceCount_ = liveVoices;
        if (muted_.load(std::memory_order_acquire)) output.fill(0);
        if (!writeAudioBlock(stop, output.data(), output.size())) break;
    }
    available_.store(false, std::memory_order_release);
}

} // namespace nec
