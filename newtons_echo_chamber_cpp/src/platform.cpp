#include "platform.hpp"

#include <X11/XKBlib.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace nec {
namespace {

using SteadyClock = std::chrono::steady_clock;

template <typename Enum>
[[nodiscard]] constexpr std::size_t enumIndex(Enum value) noexcept {
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::optional<Key> translateKey(KeySym symbol) noexcept {
    switch (symbol) {
    case XK_w:
    case XK_W:
        return Key::W;
    case XK_a:
    case XK_A:
        return Key::A;
    case XK_s:
    case XK_S:
        return Key::S;
    case XK_d:
    case XK_D:
        return Key::D;
    case XK_c:
    case XK_C:
        return Key::C;
    case XK_v:
    case XK_V:
        return Key::V;
    case XK_b:
    case XK_B:
        return Key::B;
    case XK_q:
    case XK_Q:
        return Key::Q;
    case XK_r:
    case XK_R:
        return Key::R;
    case XK_f:
    case XK_F:
        return Key::F;
    case XK_t:
    case XK_T:
        return Key::T;
    case XK_g:
    case XK_G:
        return Key::G;
    case XK_y:
    case XK_Y:
        return Key::Y;
    case XK_h:
    case XK_H:
        return Key::H;
    case XK_m:
    case XK_M:
        return Key::M;
    case XK_space:
        return Key::Space;
    case XK_Escape:
        return Key::Escape;
    case XK_Tab:
    case XK_ISO_Left_Tab:
        return Key::Tab;
    case XK_F1:
        return Key::F1;
    case XK_F2:
        return Key::F2;
    case XK_F3:
        return Key::F3;
    case XK_F4:
        return Key::F4;
    case XK_F5:
        return Key::F5;
    case XK_F6:
        return Key::F6;
    case XK_F7:
        return Key::F7;
    case XK_F8:
        return Key::F8;
    case XK_F9:
        return Key::F9;
    case XK_F10:
        return Key::F10;
    case XK_F11:
        return Key::F11;
    case XK_F12:
        return Key::F12;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] std::optional<Key> lookupKey(Display* display,
                                           XKeyEvent& event) noexcept {
    KeySym symbol = NoSymbol;
    unsigned int consumedModifiers = 0U;
    if (display != nullptr
        && XkbLookupKeySym(display, static_cast<KeyCode>(event.keycode),
                           event.state,
                           &consumedModifiers, &symbol) != False
        && symbol != NoSymbol) {
        if (const auto key = translateKey(symbol); key.has_value()) {
            return key;
        }
    }
    // Retain the base-group core-X11 lookup both for servers without usable
    // XKB state and for an alternate layout that yields an unmapped symbol.
    // This keeps the documented physical gameplay keys usable without
    // overriding a translated, explicitly supported key.
    return translateKey(XLookupKeysym(&event, 0));
}

[[nodiscard]] std::optional<MouseButton>
translateMouseButton(unsigned int button) noexcept {
    switch (button) {
    case Button1:
        return MouseButton::Left;
    case Button2:
        return MouseButton::Middle;
    case Button3:
        return MouseButton::Right;
    case 8U:
        return MouseButton::Extra1;
    case 9U:
        return MouseButton::Extra2;
    default:
        return std::nullopt;
    }
}

struct CpuCounters {
    std::uint64_t active{};
    std::uint64_t total{};
};

struct ProcSnapshot {
    // Aggregate CPU counters followed by cpu0 through cpu3.
    std::array<CpuCounters, 5> cpu{};
    std::array<bool, 5> present{};
    std::uint64_t processTicks{};
    std::uint64_t memoryTotalBytes{};
    std::uint64_t memoryAvailableBytes{};
    std::uint64_t residentBytes{};
    std::uint64_t peakResidentBytes{};
};

[[nodiscard]] bool readCpuCounters(ProcSnapshot& snapshot) {
    std::ifstream stream{"/proc/stat"};
    if (!stream) {
        return false;
    }

    std::string line;
    while (std::getline(stream, line)) {
        if (!line.starts_with("cpu")) {
            break;
        }

        std::istringstream values{line};
        std::string label;
        std::uint64_t user = 0;
        std::uint64_t nice = 0;
        std::uint64_t system = 0;
        std::uint64_t idle = 0;
        std::uint64_t ioWait = 0;
        std::uint64_t irq = 0;
        std::uint64_t softIrq = 0;
        std::uint64_t steal = 0;
        values >> label >> user >> nice >> system >> idle >> ioWait >> irq >>
            softIrq >> steal;
        if (!values) {
            continue;
        }

        std::size_t index = snapshot.cpu.size();
        if (label == "cpu") {
            index = 0;
        } else if (label.size() == 4 && label[3] >= '0' && label[3] <= '3') {
            index = 1U + static_cast<std::size_t>(label[3] - '0');
        }
        if (index >= snapshot.cpu.size()) {
            continue;
        }

        const std::uint64_t active = user + nice + system + irq + softIrq + steal;
        snapshot.cpu[index] = {active, active + idle + ioWait};
        snapshot.present[index] = true;
    }
    return snapshot.present[0];
}

[[nodiscard]] bool readProcessTicks(std::uint64_t& ticks) {
    std::ifstream stream{"/proc/self/stat"};
    std::string line;
    if (!stream || !std::getline(stream, line)) {
        return false;
    }

    // The command name in field two is parenthesized and may contain spaces.
    const std::size_t closingParenthesis = line.rfind(')');
    if (closingParenthesis == std::string::npos ||
        closingParenthesis + 2U >= line.size()) {
        return false;
    }

    std::istringstream fields{line.substr(closingParenthesis + 2U)};
    std::string ignored;
    fields >> ignored; // field 3: process state
    for (int field = 4; field <= 13; ++field) {
        fields >> ignored;
    }

    std::uint64_t userTicks = 0;
    std::uint64_t systemTicks = 0;
    fields >> userTicks >> systemTicks; // fields 14 and 15
    if (!fields) {
        return false;
    }
    ticks = userTicks + systemTicks;
    return true;
}

void readMemoryInfo(ProcSnapshot& snapshot) {
    std::ifstream memory{"/proc/meminfo"};
    std::string line;
    while (std::getline(memory, line)) {
        std::istringstream values{line};
        std::string key;
        std::uint64_t kibibytes = 0;
        values >> key >> kibibytes;
        if (!values) {
            continue;
        }
        if (key == "MemTotal:") {
            snapshot.memoryTotalBytes = kibibytes * 1024ULL;
        } else if (key == "MemAvailable:") {
            snapshot.memoryAvailableBytes = kibibytes * 1024ULL;
        }
    }

    std::ifstream status{"/proc/self/status"};
    while (std::getline(status, line)) {
        if (!line.starts_with("VmRSS:") && !line.starts_with("VmHWM:")) {
            continue;
        }
        std::istringstream values{line.substr(6U)};
        std::uint64_t kibibytes = 0;
        values >> kibibytes;
        if (!values) {
            continue;
        }
        if (line.starts_with("VmRSS:")) {
            snapshot.residentBytes = kibibytes * 1024ULL;
        } else {
            snapshot.peakResidentBytes = kibibytes * 1024ULL;
        }
    }
}

[[nodiscard]] ProcSnapshot readProcSnapshot() {
    ProcSnapshot snapshot;
    (void)readCpuCounters(snapshot);
    (void)readProcessTicks(snapshot.processTicks);
    readMemoryInfo(snapshot);
    return snapshot;
}

[[nodiscard]] float counterPercent(const CpuCounters& previous,
                                   const CpuCounters& current) noexcept {
    const std::uint64_t totalDelta = current.total >= previous.total
                                         ? current.total - previous.total
                                         : 0ULL;
    const std::uint64_t activeDelta = current.active >= previous.active
                                          ? current.active - previous.active
                                          : 0ULL;
    if (totalDelta == 0ULL) {
        return 0.0F;
    }

    const double percent = 100.0 * static_cast<double>(activeDelta) /
                           static_cast<double>(totalDelta);
    return static_cast<float>(std::clamp(percent, 0.0, 100.0));
}

} // namespace

struct Platform::Impl {
    Display* xDisplay{};
    int xScreen{};
    Window xWindow{};
    Colormap xColormap{};
    Atom wmProtocols{};
    Atom wmDeleteWindow{};
    Atom netWmName{};
    Atom utf8String{};
    Cursor hiddenCursor{};
    int width{};
    int height{};
    bool valid{};
    bool relativeRequested{};
    bool relativeActive{};
    bool detectableAutoRepeat{};
    InputState input{};
    SystemTelemetry telemetry{};
    ProcSnapshot previousProc{};
    bool haveProcBaseline{};
    SteadyClock::time_point previousTelemetryTime{};
    std::string lastError;

    void clearTransientInput() noexcept {
        input.pressed.fill(false);
        input.released.fill(false);
        input.mousePressed.fill(false);
        input.mouseReleased.fill(false);
        input.mouseDeltaX = 0.0;
        input.mouseDeltaY = 0.0;
        input.wheelDelta = 0.0;
        input.resized = false;
    }

    void releaseHeldInput() noexcept {
        for (std::size_t index = 0; index < input.held.size(); ++index) {
            if (input.held[index]) {
                input.held[index] = false;
                input.released[index] = true;
            }
        }
        for (std::size_t index = 0; index < input.mouseHeld.size(); ++index) {
            if (input.mouseHeld[index]) {
                input.mouseHeld[index] = false;
                input.mouseReleased[index] = true;
            }
        }
    }

    void deactivateRelativeMouse() noexcept {
        if (!relativeActive) {
            return;
        }
        relativeActive = false;
        if (xDisplay == nullptr || xWindow == None) {
            return;
        }
        XUngrabPointer(xDisplay, CurrentTime);
        XUndefineCursor(xDisplay, xWindow);
        XFlush(xDisplay);
    }

    [[nodiscard]] bool activateRelativeMouse() {
        if (!relativeRequested || !input.focused) {
            return true;
        }
        if (relativeActive) {
            return true;
        }
        if (!valid || xDisplay == nullptr || xWindow == None) {
            lastError = "Cannot capture the pointer without a valid X11 window";
            return false;
        }

        constexpr unsigned int eventMask = static_cast<unsigned int>(
            PointerMotionMask | ButtonPressMask | ButtonReleaseMask);
        const Cursor cursor = hiddenCursor != None ? hiddenCursor : None;
        const int result = XGrabPointer(
            xDisplay, xWindow, True, eventMask, GrabModeAsync, GrabModeAsync,
            xWindow, cursor, CurrentTime);
        if (result != GrabSuccess) {
            lastError = "XGrabPointer failed with status " +
                        std::to_string(result);
            return false;
        }

        if (hiddenCursor != None) {
            XDefineCursor(xDisplay, xWindow, hiddenCursor);
        }
        const int centerX = width / 2;
        const int centerY = height / 2;
        XWarpPointer(xDisplay, None, xWindow, 0, 0, 0U, 0U, centerX, centerY);
        XFlush(xDisplay);
        input.mouseX = centerX;
        input.mouseY = centerY;
        relativeActive = true;
        return true;
    }
};

bool InputState::down(Key key) const noexcept {
    const std::size_t index = enumIndex(key);
    return index < held.size() && held[index];
}

bool InputState::wentDown(Key key) const noexcept {
    const std::size_t index = enumIndex(key);
    return index < pressed.size() && pressed[index];
}

bool InputState::wentUp(Key key) const noexcept {
    const std::size_t index = enumIndex(key);
    return index < released.size() && released[index];
}

bool InputState::down(MouseButton button) const noexcept {
    const std::size_t index = enumIndex(button);
    return index < mouseHeld.size() && mouseHeld[index];
}

bool InputState::wentDown(MouseButton button) const noexcept {
    const std::size_t index = enumIndex(button);
    return index < mousePressed.size() && mousePressed[index];
}

bool InputState::wentUp(MouseButton button) const noexcept {
    const std::size_t index = enumIndex(button);
    return index < mouseReleased.size() && mouseReleased[index];
}

Platform::Platform() : impl_(std::make_unique<Impl>()) {}

Platform::~Platform() {
    shutdown();
}

bool Platform::create(std::string_view title, int width, int height) {
    shutdown();
    impl_->lastError.clear();
    impl_->input = {};
    impl_->telemetry = {};
    impl_->previousProc = {};
    impl_->haveProcBaseline = false;
    impl_->previousTelemetryTime = {};

    const auto fail = [this](std::string message) {
        impl_->lastError = std::move(message);
        shutdown();
        return false;
    };

    if (width <= 0 || height <= 0) {
        return fail("Window dimensions must be positive");
    }
    if (title.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return fail("Window title is too long for X11");
    }

    impl_->xDisplay = XOpenDisplay(nullptr);
    if (impl_->xDisplay == nullptr) {
        return fail("XOpenDisplay failed; DISPLAY is unavailable");
    }
    impl_->xScreen = DefaultScreen(impl_->xDisplay);

    const Window root = RootWindow(impl_->xDisplay, impl_->xScreen);
    Visual* const visual = DefaultVisual(impl_->xDisplay, impl_->xScreen);
    const int depth = DefaultDepth(impl_->xDisplay, impl_->xScreen);
    impl_->xColormap = XCreateColormap(impl_->xDisplay, root, visual, AllocNone);
    if (impl_->xColormap == None) {
        return fail("XCreateColormap failed");
    }

    XSetWindowAttributes attributes{};
    attributes.background_pixel = BlackPixel(impl_->xDisplay, impl_->xScreen);
    attributes.border_pixel = 0UL;
    attributes.colormap = impl_->xColormap;
    attributes.event_mask = StructureNotifyMask | ExposureMask | KeyPressMask |
                            KeyReleaseMask | ButtonPressMask |
                            ButtonReleaseMask | PointerMotionMask |
                            FocusChangeMask | EnterWindowMask;
    constexpr unsigned long valueMask =
        CWBackPixel | CWBorderPixel | CWColormap | CWEventMask;
    impl_->xWindow = XCreateWindow(
        impl_->xDisplay, root, 0, 0, static_cast<unsigned int>(width),
        static_cast<unsigned int>(height), 0U, depth, InputOutput, visual,
        valueMask, &attributes);
    if (impl_->xWindow == None) {
        return fail("XCreateWindow failed");
    }

    impl_->wmProtocols =
        XInternAtom(impl_->xDisplay, "WM_PROTOCOLS", False);
    impl_->wmDeleteWindow =
        XInternAtom(impl_->xDisplay, "WM_DELETE_WINDOW", False);
    if (impl_->wmProtocols == None || impl_->wmDeleteWindow == None) {
        return fail("XInternAtom failed for the WM_DELETE_WINDOW protocol");
    }
    Atom protocol = impl_->wmDeleteWindow;
    if (XSetWMProtocols(impl_->xDisplay, impl_->xWindow, &protocol, 1) == 0) {
        return fail("XSetWMProtocols failed");
    }

    impl_->netWmName = XInternAtom(impl_->xDisplay, "_NET_WM_NAME", False);
    impl_->utf8String = XInternAtom(impl_->xDisplay, "UTF8_STRING", False);

    XClassHint classHint{};
    classHint.res_name = const_cast<char*>("newtons_echo_chamber");
    classHint.res_class = const_cast<char*>("NewtonsEchoChamber");
    (void)XSetClassHint(impl_->xDisplay, impl_->xWindow, &classHint);

    // Tell both native X11 window managers and XWayland compositors that this
    // interactive window accepts keyboard focus. Focus is still acquired only
    // through normal window-manager policy or an explicit user click.
    XWMHints wmHints{};
    wmHints.flags = InputHint;
    wmHints.input = True;
    (void)XSetWMHints(impl_->xDisplay, impl_->xWindow, &wmHints);

    const Atom netWmPid =
        XInternAtom(impl_->xDisplay, "_NET_WM_PID", False);
    if (netWmPid != None) {
        const unsigned long processId = static_cast<unsigned long>(::getpid());
        (void)XChangeProperty(
            impl_->xDisplay, impl_->xWindow, netWmPid, XA_CARDINAL, 32,
            PropModeReplace,
            reinterpret_cast<const unsigned char*>(&processId), 1);
    }

    constexpr char emptyBitmap[1] = {0};
    const Pixmap emptyPixmap = XCreateBitmapFromData(
        impl_->xDisplay, impl_->xWindow, emptyBitmap, 1U, 1U);
    if (emptyPixmap != None) {
        XColor black{};
        impl_->hiddenCursor = XCreatePixmapCursor(
            impl_->xDisplay, emptyPixmap, emptyPixmap, &black, &black, 0U, 0U);
        XFreePixmap(impl_->xDisplay, emptyPixmap);
    }

    Bool detectableSupported = False;
    impl_->detectableAutoRepeat =
        XkbSetDetectableAutoRepeat(impl_->xDisplay, True,
                                   &detectableSupported) != 0 &&
        detectableSupported != False;

    impl_->width = width;
    impl_->height = height;
    impl_->input.mouseX = width / 2;
    impl_->input.mouseY = height / 2;
    impl_->input.focused = false;
    impl_->valid = true;

    if (!setTitle(title)) {
        return fail(impl_->lastError);
    }
    XMapRaised(impl_->xDisplay, impl_->xWindow);
    XFlush(impl_->xDisplay);

    (void)sampleTelemetry(true);
    return true;
}

void Platform::shutdown() noexcept {
    if (!impl_) {
        return;
    }

    impl_->deactivateRelativeMouse();
    impl_->relativeRequested = false;
    impl_->valid = false;

    if (impl_->xDisplay != nullptr) {
        if (impl_->hiddenCursor != None) {
            XFreeCursor(impl_->xDisplay, impl_->hiddenCursor);
        }
        if (impl_->xWindow != None) {
            XDestroyWindow(impl_->xDisplay, impl_->xWindow);
        }
        if (impl_->xColormap != None) {
            XFreeColormap(impl_->xDisplay, impl_->xColormap);
        }
        XCloseDisplay(impl_->xDisplay);
    }

    impl_->xDisplay = nullptr;
    impl_->xScreen = 0;
    impl_->xWindow = None;
    impl_->xColormap = None;
    impl_->wmProtocols = None;
    impl_->wmDeleteWindow = None;
    impl_->netWmName = None;
    impl_->utf8String = None;
    impl_->hiddenCursor = None;
    impl_->width = 0;
    impl_->height = 0;
    impl_->input.held.fill(false);
    impl_->input.mouseHeld.fill(false);
    impl_->input.focused = false;
}

bool Platform::pollEvents() {
    if (!impl_->valid || impl_->xDisplay == nullptr ||
        impl_->xWindow == None) {
        return false;
    }
    impl_->clearTransientInput();

    while (XPending(impl_->xDisplay) > 0) {
        XEvent event{};
        XNextEvent(impl_->xDisplay, &event);
        switch (event.type) {
        case ClientMessage:
            if (impl_->wmDeleteWindow != None &&
                event.xclient.message_type == impl_->wmProtocols &&
                event.xclient.format == 32 &&
                static_cast<Atom>(event.xclient.data.l[0]) ==
                    impl_->wmDeleteWindow) {
                impl_->input.closeRequested = true;
            }
            break;

        case DestroyNotify:
            if (event.xdestroywindow.window == impl_->xWindow) {
                impl_->input.closeRequested = true;
                impl_->input.focused = false;
                impl_->releaseHeldInput();
                // The server already destroyed this XID. Do not issue requests
                // against it from shutdown().
                impl_->relativeActive = false;
                impl_->valid = false;
                impl_->xWindow = None;
            }
            break;

        case ConfigureNotify: {
            const int newWidth = std::max(1, event.xconfigure.width);
            const int newHeight = std::max(1, event.xconfigure.height);
            if (newWidth != impl_->width || newHeight != impl_->height) {
                impl_->width = newWidth;
                impl_->height = newHeight;
                impl_->input.resized = true;
                if (impl_->relativeActive) {
                    const int centerX = newWidth / 2;
                    const int centerY = newHeight / 2;
                    XWarpPointer(impl_->xDisplay, None, impl_->xWindow, 0, 0,
                                 0U, 0U, centerX, centerY);
                    impl_->input.mouseX = centerX;
                    impl_->input.mouseY = centerY;
                }
            }
            break;
        }

        case FocusIn:
            impl_->input.focused = true;
            (void)impl_->activateRelativeMouse();
            break;

        case FocusOut:
            impl_->input.focused = false;
            impl_->releaseHeldInput();
            impl_->deactivateRelativeMouse();
            break;

        case KeyPress: {
            const auto key = lookupKey(impl_->xDisplay, event.xkey);
            if (key.has_value()) {
                const std::size_t index = enumIndex(*key);
                if (!impl_->input.held[index]) {
                    impl_->input.held[index] = true;
                    impl_->input.pressed[index] = true;
                }
            }
            break;
        }

        case KeyRelease: {
            // Without detectable repeat, X11 reports release/press pairs with
            // identical keycode and timestamp for auto-repeat. Consume the
            // paired press and retain the held state.
            if (!impl_->detectableAutoRepeat &&
                XPending(impl_->xDisplay) > 0) {
                XEvent next{};
                XPeekEvent(impl_->xDisplay, &next);
                if (next.type == KeyPress &&
                    next.xkey.keycode == event.xkey.keycode &&
                    next.xkey.time == event.xkey.time) {
                    XNextEvent(impl_->xDisplay, &next);
                    break;
                }
            }
            const auto key = lookupKey(impl_->xDisplay, event.xkey);
            if (key.has_value()) {
                const std::size_t index = enumIndex(*key);
                if (impl_->input.held[index]) {
                    impl_->input.held[index] = false;
                    impl_->input.released[index] = true;
                }
            }
            break;
        }

        case ButtonPress:
            // Some XWayland compositors deliver the click before transferring
            // keyboard focus. A real click is an explicit user gesture, so it
            // is safe to request focus using that event's server timestamp.
            if (!impl_->input.focused
                && event.xbutton.send_event == False
                && event.xbutton.window == impl_->xWindow) {
                XSetInputFocus(impl_->xDisplay, impl_->xWindow,
                               RevertToParent, event.xbutton.time);
            }
            if (event.xbutton.button == Button4) {
                impl_->input.wheelDelta += 1.0;
            } else if (event.xbutton.button == Button5) {
                impl_->input.wheelDelta -= 1.0;
            } else if (const auto button =
                           translateMouseButton(event.xbutton.button);
                       button.has_value()) {
                const std::size_t index = enumIndex(*button);
                if (!impl_->input.mouseHeld[index]) {
                    impl_->input.mouseHeld[index] = true;
                    impl_->input.mousePressed[index] = true;
                }
            }
            break;

        case ButtonRelease:
            if (const auto button =
                    translateMouseButton(event.xbutton.button);
                button.has_value()) {
                const std::size_t index = enumIndex(*button);
                if (impl_->input.mouseHeld[index]) {
                    impl_->input.mouseHeld[index] = false;
                    impl_->input.mouseReleased[index] = true;
                }
            }
            break;

        case MotionNotify:
            if (!impl_->relativeActive) {
                impl_->input.mouseDeltaX += static_cast<double>(
                    event.xmotion.x - impl_->input.mouseX);
                impl_->input.mouseDeltaY += static_cast<double>(
                    event.xmotion.y - impl_->input.mouseY);
                impl_->input.mouseX = event.xmotion.x;
                impl_->input.mouseY = event.xmotion.y;
            }
            break;

        case EnterNotify:
            if (!impl_->relativeActive) {
                impl_->input.mouseX = event.xcrossing.x;
                impl_->input.mouseY = event.xcrossing.y;
            }
            break;

        default:
            break;
        }
    }

    if (impl_->relativeActive && impl_->valid &&
        !impl_->input.closeRequested) {
        Window rootReturn = None;
        Window childReturn = None;
        int rootX = 0;
        int rootY = 0;
        int windowX = 0;
        int windowY = 0;
        unsigned int mask = 0U;
        if (XQueryPointer(impl_->xDisplay, impl_->xWindow, &rootReturn,
                          &childReturn, &rootX, &rootY, &windowX, &windowY,
                          &mask) != False) {
            const int centerX = impl_->width / 2;
            const int centerY = impl_->height / 2;
            impl_->input.mouseDeltaX +=
                static_cast<double>(windowX - centerX);
            impl_->input.mouseDeltaY +=
                static_cast<double>(windowY - centerY);
            impl_->input.mouseX = centerX;
            impl_->input.mouseY = centerY;
            if (windowX != centerX || windowY != centerY) {
                XWarpPointer(impl_->xDisplay, None, impl_->xWindow, 0, 0, 0U,
                             0U, centerX, centerY);
                XFlush(impl_->xDisplay);
            }
        }
    }

    (void)sampleTelemetry(false);
    return impl_->valid && !impl_->input.closeRequested;
}

const InputState& Platform::input() const noexcept {
    return impl_->input;
}

bool Platform::closeRequested() const noexcept {
    return impl_->input.closeRequested;
}

void Platform::requestClose() noexcept {
    impl_->input.closeRequested = true;
}

bool Platform::setRelativeMouse(bool enabled) {
    if (!impl_->valid || impl_->xDisplay == nullptr ||
        impl_->xWindow == None) {
        impl_->lastError =
            "Cannot change mouse capture before creating the platform";
        return false;
    }

    impl_->relativeRequested = enabled;
    if (!enabled) {
        impl_->deactivateRelativeMouse();
        return true;
    }
    return impl_->activateRelativeMouse();
}

bool Platform::relativeMouse() const noexcept {
    return impl_->relativeActive;
}

int Platform::width() const noexcept {
    return impl_->width;
}

int Platform::height() const noexcept {
    return impl_->height;
}

bool Platform::valid() const noexcept {
    return impl_->valid;
}

std::string_view Platform::lastError() const noexcept {
    return impl_->lastError;
}

bool Platform::setTitle(std::string_view title) {
    if (!impl_->valid || impl_->xDisplay == nullptr ||
        impl_->xWindow == None) {
        impl_->lastError = "Cannot set a title without a valid X11 window";
        return false;
    }
    if (title.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        impl_->lastError = "Window title is too long for X11";
        return false;
    }

    const std::string ownedTitle{title};
    if (XStoreName(impl_->xDisplay, impl_->xWindow,
                   ownedTitle.c_str()) == 0) {
        impl_->lastError = "XStoreName failed";
        return false;
    }

    if (impl_->netWmName != None && impl_->utf8String != None) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(
            ownedTitle.data());
        (void)XChangeProperty(
            impl_->xDisplay, impl_->xWindow, impl_->netWmName,
            impl_->utf8String, 8, PropModeReplace, bytes,
            static_cast<int>(ownedTitle.size()));
    }
    XFlush(impl_->xDisplay);
    return true;
}

void* Platform::nativeDisplay() const noexcept {
    return static_cast<void*>(impl_->xDisplay);
}

unsigned long Platform::nativeWindow() const noexcept {
    return static_cast<unsigned long>(impl_->xWindow);
}

bool Platform::sampleTelemetry(bool force) {
    const auto now = SteadyClock::now();
    if (!force && impl_->haveProcBaseline &&
        now - impl_->previousTelemetryTime < std::chrono::milliseconds{500}) {
        return true;
    }

    const ProcSnapshot current = readProcSnapshot();
    if (!current.present[0]) {
        return false;
    }

    const long online = ::sysconf(_SC_NPROCESSORS_ONLN);
    impl_->telemetry.onlineCpuCount =
        online > 0L ? static_cast<unsigned int>(online) : 0U;
    impl_->telemetry.physicalMemoryBytes = current.memoryTotalBytes;
    impl_->telemetry.availableMemoryBytes = current.memoryAvailableBytes;
    impl_->telemetry.processResidentBytes = current.residentBytes;
    impl_->telemetry.processPeakResidentBytes = current.peakResidentBytes;

    if (impl_->haveProcBaseline) {
        impl_->telemetry.totalCpuUsagePercent =
            counterPercent(impl_->previousProc.cpu[0], current.cpu[0]);
        for (std::size_t core = 0;
             core < impl_->telemetry.coreUsagePercent.size(); ++core) {
            const std::size_t counter = core + 1U;
            impl_->telemetry.coreUsagePercent[core] =
                current.present[counter] && impl_->previousProc.present[counter]
                    ? counterPercent(impl_->previousProc.cpu[counter],
                                     current.cpu[counter])
                    : 0.0F;
        }

        const double elapsed = std::chrono::duration<double>(
                                   now - impl_->previousTelemetryTime)
                                   .count();
        const long ticksPerSecond = ::sysconf(_SC_CLK_TCK);
        if (elapsed > 0.0 && ticksPerSecond > 0L &&
            current.processTicks >= impl_->previousProc.processTicks) {
            const double tickDelta = static_cast<double>(
                current.processTicks - impl_->previousProc.processTicks);
            const double processPercent =
                100.0 * tickDelta / static_cast<double>(ticksPerSecond) /
                elapsed;
            const double upperBound =
                100.0 * static_cast<double>(
                            std::max(1U, impl_->telemetry.onlineCpuCount));
            impl_->telemetry.processCpuUsagePercent = static_cast<float>(
                std::clamp(processPercent, 0.0, upperBound));
        }
    }

    impl_->previousProc = current;
    impl_->previousTelemetryTime = now;
    impl_->haveProcBaseline = true;
    ++impl_->telemetry.sampleSequence;
    return true;
}

const SystemTelemetry& Platform::telemetry() const noexcept {
    return impl_->telemetry;
}

} // namespace nec
