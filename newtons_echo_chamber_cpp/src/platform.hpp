#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace nec {

enum class Key : std::uint8_t {
    W,
    A,
    S,
    D,
    C,
    V,
    B,
    Q,
    R,
    F,
    T,
    G,
    Y,
    H,
    M,
    Space,
    Escape,
    Tab,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Count,
};

enum class MouseButton : std::uint8_t {
    Left,
    Middle,
    Right,
    Extra1,
    Extra2,
    Count,
};

inline constexpr std::size_t kKeyCount = static_cast<std::size_t>(Key::Count);
inline constexpr std::size_t kMouseButtonCount =
    static_cast<std::size_t>(MouseButton::Count);

struct InputState {
    std::array<bool, kKeyCount> held{};
    std::array<bool, kKeyCount> pressed{};
    std::array<bool, kKeyCount> released{};
    std::array<bool, kMouseButtonCount> mouseHeld{};
    std::array<bool, kMouseButtonCount> mousePressed{};
    std::array<bool, kMouseButtonCount> mouseReleased{};
    double mouseDeltaX{};
    double mouseDeltaY{};
    double wheelDelta{};
    int mouseX{};
    int mouseY{};
    bool focused{};
    bool resized{};
    bool closeRequested{};

    [[nodiscard]] bool down(Key key) const noexcept;
    [[nodiscard]] bool wentDown(Key key) const noexcept;
    [[nodiscard]] bool wentUp(Key key) const noexcept;
    [[nodiscard]] bool down(MouseButton button) const noexcept;
    [[nodiscard]] bool wentDown(MouseButton button) const noexcept;
    [[nodiscard]] bool wentUp(MouseButton button) const noexcept;
};

struct SystemTelemetry {
    std::array<float, 4> coreUsagePercent{};
    float totalCpuUsagePercent{};
    // Process CPU can reach 400 percent when all Pi 5 cores are saturated.
    float processCpuUsagePercent{};
    unsigned onlineCpuCount{};
    std::uint64_t physicalMemoryBytes{};
    std::uint64_t availableMemoryBytes{};
    std::uint64_t processResidentBytes{};
    std::uint64_t processPeakResidentBytes{};
    std::uint64_t sampleSequence{};
};

// Owns only the X11 window and the per-frame input snapshot. Vulkan instance,
// surface, swapchain, presentation and synchronization remain renderer-owned.
// Methods that touch X11 (create, shutdown, pollEvents, setRelativeMouse and
// setTitle) must be called from the same main thread.
class Platform final {
public:
    Platform();
    ~Platform();

    Platform(const Platform&) = delete;
    Platform& operator=(const Platform&) = delete;
    Platform(Platform&&) = delete;
    Platform& operator=(Platform&&) = delete;

    [[nodiscard]] bool create(std::string_view title, int width, int height);
    void shutdown() noexcept;

    // Clears transient input and drains the X11 queue. Returns false once a
    // close is requested or the native window has been destroyed.
    [[nodiscard]] bool pollEvents();
    [[nodiscard]] const InputState& input() const noexcept;
    [[nodiscard]] bool closeRequested() const noexcept;
    void requestClose() noexcept;

    // Relative capture uses a confined, invisible pointer and center warping,
    // avoiding a dependency on the XInput2 development package.
    [[nodiscard]] bool setRelativeMouse(bool enabled);
    [[nodiscard]] bool relativeMouse() const noexcept;

    [[nodiscard]] int width() const noexcept;
    [[nodiscard]] int height() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::string_view lastError() const noexcept;

    // Changes both the legacy WM_NAME and UTF-8 _NET_WM_NAME properties.
    [[nodiscard]] bool setTitle(std::string_view title);

    // Native handles for VkXlibSurfaceCreateInfoKHR. They remain owned by this
    // object and are valid only between successful create() and shutdown().
    [[nodiscard]] void* nativeDisplay() const noexcept;
    [[nodiscard]] unsigned long nativeWindow() const noexcept;

    // Samples /proc at most twice per second unless force is true. The first
    // sample establishes CPU counters; later samples provide percentages.
    [[nodiscard]] bool sampleTelemetry(bool force = false);
    [[nodiscard]] const SystemTelemetry& telemetry() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nec
