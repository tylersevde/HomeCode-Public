#pragma once

#include "core.hpp"

#include <array>

namespace nec {

class AudioEngine {
public:
    explicit AudioEngine(bool enabled = true);
    ~AudioEngine();
    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    bool play(std::string_view key, float volume = 1.0F, float pan = 0.0F);
    bool playImpact(const ImpactEvent& event, const Player& listener);
    void toggleMute() noexcept;
    [[nodiscard]] bool muted() const noexcept {
        return muted_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool available() const noexcept {
        return available_.load(std::memory_order_acquire);
    }

private:
    static constexpr std::size_t kRequestCapacity = 64U;
    static constexpr std::size_t kVoiceCapacity = 24U;
    static constexpr std::size_t kWallCount = 6U;
    static constexpr std::size_t kEarlyDelayCapacity = 4096U;
    static constexpr std::size_t kReverbLineCount = 4U;
    static constexpr std::size_t kReverbDelayCapacity = 2048U;

    struct VoiceRequest {
        const std::vector<std::int16_t>* samples{};
        float left{};
        float right{};
        float roomSend{};
        std::uint64_t muteGeneration{};
    };
    struct Voice {
        const std::vector<std::int16_t>* samples{};
        std::size_t frame{};
        float left{};
        float right{};
        float roomSend{};
    };
    struct StereoFrame {
        float left{};
        float right{};
    };

    void buildBank();
    void audioLoop(std::stop_token stop);
    void resetMixerState() noexcept;
    [[nodiscard]] StereoFrame processRoom(float input) noexcept;
    [[nodiscard]] bool startTransport() noexcept;
    void stopTransport() noexcept;
    [[nodiscard]] bool writeAudioBlock(std::stop_token stop,
                                       const std::int16_t* samples,
                                       std::size_t sampleCount) noexcept;
    [[nodiscard]] static std::vector<std::int16_t> synthesize(
        float seconds, const std::function<float(float, std::size_t)>& sample);

    std::unordered_map<std::string, std::vector<std::int16_t>> bank_;
    std::mutex queueMutex_;
    std::array<VoiceRequest, kRequestCapacity> requests_{};
    std::size_t requestHead_{};
    std::size_t requestCount_{};
    std::array<Voice, kVoiceCapacity> voices_{};
    std::size_t voiceCount_{};

    // The effect state is owned exclusively by the audio thread. Power-of-two
    // storage keeps the six early-reflection taps cheap enough for Pi-class
    // hardware, while the four unequal delay lengths form a compact FDN.
    std::array<float, kEarlyDelayCapacity> earlyDelay_{};
    std::array<float, kWallCount> earlyLowPass_{};
    std::size_t earlyCursor_{};
    std::array<std::array<float, kReverbDelayCapacity>, kReverbLineCount>
        reverbDelay_{};
    std::array<std::size_t, kReverbLineCount> reverbCursor_{};
    std::array<float, kReverbLineCount> reverbLowPass_{};
    float roomDcInput_{};
    float roomDcOutput_{};
    std::uint64_t mixerMuteGeneration_{};

    std::atomic<bool> muted_{false};
    // Incremented only on an unmuted -> muted transition. Requests carry the
    // generation they were created in, so an event racing a mute can never
    // reappear when the game is unmuted.
    std::atomic<std::uint64_t> muteGeneration_{0U};
    std::atomic<bool> available_{false};
    int writeFd_{-1};
    int childPid_{-1};
    std::jthread thread_;
};

} // namespace nec
