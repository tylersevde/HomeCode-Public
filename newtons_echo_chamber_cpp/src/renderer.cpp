#include "renderer.hpp"

#include "platform.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef NEC_SHADER_BINARY_DIR
#define NEC_SHADER_BINARY_DIR "."
#endif
#ifndef NEC_INSTALL_SHADER_DIR
#define NEC_INSTALL_SHADER_DIR "."
#endif
#ifndef NEC_TEXTURE_ASSET_DIR
#define NEC_TEXTURE_ASSET_DIR "."
#endif
#ifndef NEC_INSTALL_TEXTURE_DIR
#define NEC_INSTALL_TEXTURE_DIR "."
#endif

namespace nec {
namespace {

constexpr std::uint32_t kFrameCount = 2U;
constexpr std::uint32_t kTimestampTotalStart = 0U;
constexpr std::uint32_t kTimestampComputeStart = 1U;
constexpr std::uint32_t kTimestampComputeEnd = 2U;
constexpr std::uint32_t kTimestampShadowStart = 3U;
constexpr std::uint32_t kTimestampShadowEnd = 4U;
constexpr std::uint32_t kTimestampGraphicsStart = 5U;
constexpr std::uint32_t kTimestampGraphicsEnd = 6U;
constexpr std::uint32_t kTimestampQueryCount = 7U;
constexpr std::uint32_t kShadowResolution = 1'024U;
constexpr std::array<std::uint32_t, 3> kShadowQualityResolution{
    512U, 768U, kShadowResolution};
constexpr std::uint32_t kMaterialAtlasExtent = 1'024U;
constexpr std::uint32_t kMaterialAtlasColumns = 4U;
constexpr std::uint32_t kMaterialLayerExtent = 256U;
constexpr std::uint32_t kMaterialLayerCount = 16U;
constexpr std::uint32_t kMaterialMipLevels = 9U;
// Normal detail is deliberately bounded on the fragment path.  Near fragments
// receive the authored tangent-space normal, the following interval fades back
// to the geometric normal, and fragments beyond the cutoff skip the texture
// fetch entirely.  These ranges keep SAFE economical on V3D while ULTRA retains
// room-scale detail across a much larger part of the chamber.
constexpr std::array<float, 3> kNormalDetailFullRange{{8.0F, 13.0F, 20.0F}};
constexpr std::array<float, 3> kNormalDetailCutoff{{15.0F, 24.0F, 38.0F}};
constexpr std::uint32_t kParticleLocalSize = 128U;
constexpr std::uint32_t kSafeParticles = 32'768U;
constexpr std::uint32_t kBalancedParticles = 98'304U;
constexpr std::uint32_t kUltraParticles = 196'608U;
constexpr std::uint32_t kMaximumParticles = 262'144U;
constexpr std::size_t kInstanceCapacity = 65'536U;
constexpr std::uint32_t kImpactCommandCapacity = 24U;
constexpr std::uint32_t kImpactEmissionCapacity = 2'048U;
constexpr std::size_t kResonancePulseCapacity = 12U;
constexpr std::size_t kEchoPulseVisualCapacity = 4U;
constexpr std::size_t kEchoPulseGlyphCapacity = 64U;
constexpr std::size_t kHudVertexCapacity = 131'072U;

[[nodiscard]] float gpu(Real value) noexcept {
    return static_cast<float>(value);
}

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

[[nodiscard]] std::string lowerCopy(std::string_view value) {
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(), [](char character) {
        const unsigned char byte = static_cast<unsigned char>(character);
        return static_cast<char>(byte >= static_cast<unsigned char>('A')
                              && byte <= static_cast<unsigned char>('Z')
            ? byte - static_cast<unsigned char>('A') + static_cast<unsigned char>('a')
            : byte);
    });
    return result;
}

[[nodiscard]] bool contains(std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
}

[[nodiscard]] bool isV3dv(std::string_view deviceName) {
    const std::string name = lowerCopy(deviceName);
    return contains(name, "v3dv") || contains(name, "v3d ")
        || contains(name, "v3d-") || contains(name, "videocore vii")
        || contains(name, "videocore 7");
}

[[nodiscard]] bool isSoftwareDevice(const VkPhysicalDeviceProperties& properties) {
    const std::string name = lowerCopy(properties.deviceName);
    return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU
        || contains(name, "llvmpipe") || contains(name, "lavapipe")
        || contains(name, "swiftshader") || contains(name, "software rasterizer");
}

[[nodiscard]] bool hasName(const std::vector<VkExtensionProperties>& values,
                           const char* name) {
    return std::any_of(values.begin(), values.end(), [name](const auto& value) {
        return std::strcmp(value.extensionName, name) == 0;
    });
}

[[nodiscard]] bool hasName(const std::vector<VkLayerProperties>& values,
                           const char* name) {
    return std::any_of(values.begin(), values.end(), [name](const auto& value) {
        return std::strcmp(value.layerName, name) == 0;
    });
}

[[nodiscard]] std::vector<VkExtensionProperties>
instanceExtensions() {
    std::uint32_t count = 0U;
    requireVk(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
              "vkEnumerateInstanceExtensionProperties(count)");
    std::vector<VkExtensionProperties> values(count);
    requireVk(vkEnumerateInstanceExtensionProperties(nullptr, &count, values.data()),
              "vkEnumerateInstanceExtensionProperties");
    values.resize(count);
    return values;
}

[[nodiscard]] std::vector<VkLayerProperties> instanceLayers() {
    std::uint32_t count = 0U;
    requireVk(vkEnumerateInstanceLayerProperties(&count, nullptr),
              "vkEnumerateInstanceLayerProperties(count)");
    std::vector<VkLayerProperties> values(count);
    requireVk(vkEnumerateInstanceLayerProperties(&count, values.data()),
              "vkEnumerateInstanceLayerProperties");
    values.resize(count);
    return values;
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

struct Mat4 {
    std::array<float, 16> value{};
};

[[nodiscard]] Mat4 multiply(const Mat4& left, const Mat4& right) noexcept {
    Mat4 result{};
    for (std::size_t column = 0U; column < 4U; ++column) {
        for (std::size_t row = 0U; row < 4U; ++row) {
            float sum = 0.0F;
            for (std::size_t index = 0U; index < 4U; ++index) {
                sum += left.value[index * 4U + row]
                     * right.value[column * 4U + index];
            }
            result.value[column * 4U + row] = sum;
        }
    }
    return result;
}

[[nodiscard]] Mat4 perspective(float verticalRadians, float aspect,
                               float nearPlane, float farPlane) noexcept {
    const float scale = 1.0F / std::tan(verticalRadians * 0.5F);
    Mat4 result{};
    result.value[0] = scale / std::max(aspect, 0.01F);
    // Vulkan's framebuffer Y axis points down when a positive viewport height
    // is used, so flip Y in the projection rather than requiring maintenance1.
    result.value[5] = -scale;
    result.value[10] = farPlane / (nearPlane - farPlane);
    result.value[11] = -1.0F;
    result.value[14] = (farPlane * nearPlane) / (nearPlane - farPlane);
    return result;
}

[[nodiscard]] Mat4 lookAt(const Vec3& eye, const Vec3& target) noexcept {
    const Vec3 forward = (target - eye).normalized({0.0, 0.0, 1.0});
    const Vec3 side = forward.cross({0.0, 1.0, 0.0}).normalized({1.0, 0.0, 0.0});
    const Vec3 up = side.cross(forward);
    return {{
        gpu(side.x), gpu(up.x), gpu(-forward.x), 0.0F,
        gpu(side.y), gpu(up.y), gpu(-forward.y), 0.0F,
        gpu(side.z), gpu(up.z), gpu(-forward.z), 0.0F,
        gpu(-side.dot(eye)), gpu(-up.dot(eye)), gpu(forward.dot(eye)), 1.0F,
    }};
}

[[nodiscard]] Quat interpolateQuaternion(const Quat& from, const Quat& to,
                                         float amount) noexcept {
    const Real alignment = from.w * to.w + from.x * to.x
                         + from.y * to.y + from.z * to.z;
    const Real sign = alignment < 0.0 ? -1.0 : 1.0;
    const Real alpha = static_cast<Real>(clamp(amount, 0.0F, 1.0F));
    return Quat{
        from.w * (1.0 - alpha) + to.w * sign * alpha,
        from.x * (1.0 - alpha) + to.x * sign * alpha,
        from.y * (1.0 - alpha) + to.y * sign * alpha,
        from.z * (1.0 - alpha) + to.z * sign * alpha,
    }.normalized();
}

// Builds the shortest rotation from a cube's local +X axis to a world-space
// vector.  Echo Pulse velocity glyphs use this to remain a single inexpensive
// instanced cube rather than introducing a line/geometry pipeline.
[[nodiscard]] Quat orientXAxis(const Vec3& vector) noexcept {
    const Vec3 direction = vector.normalized({1.0, 0.0, 0.0});
    const Real alignment = clamp(direction.x, -1.0, 1.0);
    if (alignment < -0.999999) {
        return {0.0, 0.0, 1.0, 0.0};
    }
    return Quat{1.0 + alignment, 0.0, -direction.z, direction.y}.normalized();
}

struct Vertex {
    float position[3]{};
    float normal[3]{};
    float tangent[4]{};
    float textureCoordinate[2]{};
};
static_assert(sizeof(Vertex) == 48U);

struct alignas(16) Instance {
    float positionScaleX[4]{};
    float quaternion[4]{};
    float scaleYZMaterial[4]{};
    float color[4]{};
};
static_assert(sizeof(Instance) == 64U);

struct alignas(16) Particle {
    float positionLife[4]{};
    float velocitySeed[4]{};
};
static_assert(sizeof(Particle) == 32U);

struct alignas(16) ImpactCommand {
    // xyz: world-space origin, w: visual strength.
    float positionStrength[4]{};
    // x: material family, y: first emitted particle, z: emission count,
    // w: monotonically changing frame serial.
    std::uint32_t metadata[4]{};
};
static_assert(sizeof(ImpactCommand) == 32U);

struct ResonancePulse {
    Vec3 position{};
    double startTime{-1.0};
    float strength{};
    std::uint32_t family{};
};

struct EchoPulseGlyphSnapshot {
    std::uint16_t body{};
    Vec3 position{};
    Vec3 impulse{};
};

struct EchoPulseVisual {
    std::uint64_t serial{};
    Vec3 origin{};
    double startTime{-1.0};
    Real influenceRadius{};
    std::uint32_t affectedBodyCount{};
    Real totalDeliveredImpulse{};
    std::array<EchoPulseGlyphSnapshot, kEchoPulseGlyphCapacity> glyphs{};
    std::size_t glyphCount{};
};

struct HudVertex {
    float position[2]{};
    float color[4]{};
};
static_assert(sizeof(HudVertex) == 24U);

struct alignas(16) FrameBlock {
    std::array<float, 16> viewProjection{};
    std::array<float, 4> cameraFogStart{};
    std::array<float, 4> fogEndAmbientCountTime{};
    std::array<std::array<float, 4>, 4> lightPosition{};
    std::array<std::array<float, 4>, 4> lightColor{};
    std::array<float, 4> flashlightPositionEnabled{};
    std::array<float, 4> flashlightDirectionRange{};
    std::array<float, 16> flashlightViewProjection{};
    std::array<float, 4> flashlightShadowParams{};
    std::array<float, 4> materialOptions{};
};
static_assert(sizeof(FrameBlock) == 352U);

struct SimulationPush {
    std::uint32_t count{};
    float deltaTime{};
    float gravity{};
    float friction{};
    float elapsedTime{};
    std::uint32_t impactCount{};
    std::uint32_t impactEmissionCount{};
    std::uint32_t impactFirstParticle{};
    std::uint32_t impactEndParticle{};
    std::uint32_t frameSerial{};
};
static_assert(sizeof(SimulationPush) == 40U);

using HudColor = std::array<float, 4>;

[[nodiscard]] std::array<std::uint8_t, 7> glyphRows(char character) noexcept {
    if (character >= 'a' && character <= 'z') {
        character = static_cast<char>(character - 'a' + 'A');
    }
    using Rows = std::array<std::uint8_t, 7>;
    switch (character) {
    case 'A': return Rows{0x0e,0x11,0x11,0x1f,0x11,0x11,0x11};
    case 'B': return Rows{0x1e,0x11,0x11,0x1e,0x11,0x11,0x1e};
    case 'C': return Rows{0x0f,0x10,0x10,0x10,0x10,0x10,0x0f};
    case 'D': return Rows{0x1e,0x11,0x11,0x11,0x11,0x11,0x1e};
    case 'E': return Rows{0x1f,0x10,0x10,0x1e,0x10,0x10,0x1f};
    case 'F': return Rows{0x1f,0x10,0x10,0x1e,0x10,0x10,0x10};
    case 'G': return Rows{0x0f,0x10,0x10,0x13,0x11,0x11,0x0f};
    case 'H': return Rows{0x11,0x11,0x11,0x1f,0x11,0x11,0x11};
    case 'I': return Rows{0x1f,0x04,0x04,0x04,0x04,0x04,0x1f};
    case 'J': return Rows{0x01,0x01,0x01,0x01,0x11,0x11,0x0e};
    case 'K': return Rows{0x11,0x12,0x14,0x18,0x14,0x12,0x11};
    case 'L': return Rows{0x10,0x10,0x10,0x10,0x10,0x10,0x1f};
    case 'M': return Rows{0x11,0x1b,0x15,0x15,0x11,0x11,0x11};
    case 'N': return Rows{0x11,0x19,0x15,0x13,0x11,0x11,0x11};
    case 'O': return Rows{0x0e,0x11,0x11,0x11,0x11,0x11,0x0e};
    case 'P': return Rows{0x1e,0x11,0x11,0x1e,0x10,0x10,0x10};
    case 'Q': return Rows{0x0e,0x11,0x11,0x11,0x15,0x12,0x0d};
    case 'R': return Rows{0x1e,0x11,0x11,0x1e,0x14,0x12,0x11};
    case 'S': return Rows{0x0f,0x10,0x10,0x0e,0x01,0x01,0x1e};
    case 'T': return Rows{0x1f,0x04,0x04,0x04,0x04,0x04,0x04};
    case 'U': return Rows{0x11,0x11,0x11,0x11,0x11,0x11,0x0e};
    case 'V': return Rows{0x11,0x11,0x11,0x11,0x11,0x0a,0x04};
    case 'W': return Rows{0x11,0x11,0x11,0x15,0x15,0x15,0x0a};
    case 'X': return Rows{0x11,0x11,0x0a,0x04,0x0a,0x11,0x11};
    case 'Y': return Rows{0x11,0x11,0x0a,0x04,0x04,0x04,0x04};
    case 'Z': return Rows{0x1f,0x01,0x02,0x04,0x08,0x10,0x1f};
    case '0': return Rows{0x0e,0x11,0x13,0x15,0x19,0x11,0x0e};
    case '1': return Rows{0x04,0x0c,0x04,0x04,0x04,0x04,0x0e};
    case '2': return Rows{0x0e,0x11,0x01,0x02,0x04,0x08,0x1f};
    case '3': return Rows{0x1e,0x01,0x01,0x0e,0x01,0x01,0x1e};
    case '4': return Rows{0x02,0x06,0x0a,0x12,0x1f,0x02,0x02};
    case '5': return Rows{0x1f,0x10,0x10,0x1e,0x01,0x01,0x1e};
    case '6': return Rows{0x0e,0x10,0x10,0x1e,0x11,0x11,0x0e};
    case '7': return Rows{0x1f,0x01,0x02,0x04,0x08,0x08,0x08};
    case '8': return Rows{0x0e,0x11,0x11,0x0e,0x11,0x11,0x0e};
    case '9': return Rows{0x0e,0x11,0x11,0x0f,0x01,0x01,0x0e};
    case '.': return Rows{0,0,0,0,0,0x06,0x06};
    case ',': return Rows{0,0,0,0,0x06,0x06,0x04};
    case ':': return Rows{0,0x06,0x06,0,0x06,0x06,0};
    case ';': return Rows{0,0x06,0x06,0,0x06,0x04,0x08};
    case '!': return Rows{0x04,0x04,0x04,0x04,0x04,0,0x04};
    case '?': return Rows{0x0e,0x11,0x01,0x02,0x04,0,0x04};
    case '-': return Rows{0,0,0,0x1f,0,0,0};
    case '+': return Rows{0,0x04,0x04,0x1f,0x04,0x04,0};
    case '/': return Rows{0x01,0x02,0x02,0x04,0x08,0x08,0x10};
    case '\\': return Rows{0x10,0x08,0x08,0x04,0x02,0x02,0x01};
    case '(': return Rows{0x02,0x04,0x08,0x08,0x08,0x04,0x02};
    case ')': return Rows{0x08,0x04,0x02,0x02,0x02,0x04,0x08};
    case '[': return Rows{0x0e,0x08,0x08,0x08,0x08,0x08,0x0e};
    case ']': return Rows{0x0e,0x02,0x02,0x02,0x02,0x02,0x0e};
    case '=': return Rows{0,0x1f,0,0x1f,0,0,0};
    case '_': return Rows{0,0,0,0,0,0,0x1f};
    case '%': return Rows{0x19,0x1a,0x02,0x04,0x08,0x0b,0x13};
    case '#': return Rows{0x0a,0x1f,0x0a,0x0a,0x1f,0x0a,0};
    case '<': return Rows{0x02,0x04,0x08,0x10,0x08,0x04,0x02};
    case '>': return Rows{0x08,0x04,0x02,0x01,0x02,0x04,0x08};
    case '\'': return Rows{0x04,0x04,0x08,0,0,0,0};
    case '"': return Rows{0x0a,0x0a,0x0a,0,0,0,0};
    case '*': return Rows{0,0x15,0x0e,0x1f,0x0e,0x15,0};
    case ' ': return Rows{};
    default: return Rows{0x0e,0x11,0x01,0x02,0x04,0,0x04};
    }
}

struct HudBuilder {
    std::vector<HudVertex>& vertices;
    float width{};
    float height{};

    void quad(float x, float y, float quadWidth, float quadHeight,
              const HudColor& color) {
        if (quadWidth <= 0.0F || quadHeight <= 0.0F
            || vertices.size() + 6U > kHudVertexCapacity) {
            return;
        }
        const float left = x * 2.0F / width - 1.0F;
        const float right = (x + quadWidth) * 2.0F / width - 1.0F;
        // A positive Vulkan viewport maps NDC -Y to the framebuffer's top.
        // Scene projection flips Y for camera convention; HUD coordinates are
        // already expressed from the top-left and map directly here.
        const float top = y * 2.0F / height - 1.0F;
        const float bottom = (y + quadHeight) * 2.0F / height - 1.0F;
        const auto vertex = [&](float px, float py) {
            return HudVertex{{px, py},
                {color[0], color[1], color[2], color[3]}};
        };
        vertices.push_back(vertex(left, top));
        vertices.push_back(vertex(left, bottom));
        vertices.push_back(vertex(right, bottom));
        vertices.push_back(vertex(left, top));
        vertices.push_back(vertex(right, bottom));
        vertices.push_back(vertex(right, top));
    }

    void text(float x, float y, std::string_view value, float scale,
              const HudColor& color) {
        const float originX = x;
        for (char character : value) {
            if (character == '\n') {
                x = originX;
                y += scale * 9.0F;
                continue;
            }
            const auto rows = glyphRows(character);
            for (std::size_t row = 0U; row < rows.size(); ++row) {
                std::uint8_t bits = rows[row];
                int column = 0;
                while (column < 5) {
                    while (column < 5
                           && (bits & static_cast<std::uint8_t>(1U << (4 - column)))
                                  == 0U) {
                        ++column;
                    }
                    const int first = column;
                    while (column < 5
                           && (bits & static_cast<std::uint8_t>(1U << (4 - column)))
                                  != 0U) {
                        ++column;
                    }
                    if (first < column) {
                        quad(x + static_cast<float>(first) * scale,
                             y + static_cast<float>(row) * scale,
                             static_cast<float>(column - first) * scale,
                             scale, color);
                    }
                }
            }
            x += scale * 6.0F;
        }
    }
};

[[nodiscard]]
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
std::string hudFormat(const char* format, ...) {
    std::array<char, 512> buffer{};
    std::va_list arguments;
    va_start(arguments, format);
    const int count = std::vsnprintf(
        buffer.data(), buffer.size(), format, arguments);
    va_end(arguments);
    if (count <= 0) {
        return {};
    }
    return std::string(buffer.data(), std::min<std::size_t>(
        static_cast<std::size_t>(count), buffer.size() - 1U));
}

[[nodiscard]] std::string clippedHudText(std::string_view value,
                                         std::size_t maximum) {
    if (value.size() <= maximum) {
        return std::string(value);
    }
    if (maximum <= 3U) {
        return std::string(value.substr(0U, maximum));
    }
    std::string result(value.substr(0U, maximum - 3U));
    result += "...";
    return result;
}

[[nodiscard]] std::string_view materialName(SurfaceMaterial material) noexcept {
    switch (material) {
    case SurfaceMaterial::Concrete: return "CONCRETE";
    case SurfaceMaterial::PaintedConcrete: return "PAINTED CONCRETE";
    case SurfaceMaterial::Rubber: return "RUBBER";
    case SurfaceMaterial::Metal: return "METAL";
    case SurfaceMaterial::Wood: return "WOOD";
    case SurfaceMaterial::Latex: return "LATEX";
    case SurfaceMaterial::Foam: return "FOAM";
    case SurfaceMaterial::Goo: return "GOO";
    case SurfaceMaterial::Ceramic: return "CERAMIC";
    case SurfaceMaterial::Clay: return "CLAY";
    case SurfaceMaterial::Bowling: return "BOWLING URETHANE";
    case SurfaceMaterial::Plush: return "PLUSH";
    case SurfaceMaterial::Neutral: return "NEUTRAL";
    }
    return "NEUTRAL";
}

[[nodiscard]] std::uint32_t impactFamily(std::string_view family) noexcept {
    if (contains(family, "steel") || contains(family, "wheelbarrow")) return 1U;
    if (contains(family, "concrete") || contains(family, "ceramic")
        || contains(family, "clay")) return 2U;
    if (contains(family, "wood") || contains(family, "bat")
        || contains(family, "pallet") || contains(family, "timber")) return 3U;
    if (contains(family, "rubber") || contains(family, "dodge")
        || contains(family, "medicine") || contains(family, "bowling")) return 4U;
    if (contains(family, "goo")) return 5U;
    if (contains(family, "plush")) return 6U;
    if (contains(family, "balloon") || contains(family, "latex")) return 7U;
    if (contains(family, "foam") || contains(family, "noodle")) return 8U;
    return 0U;
}

[[nodiscard]] Vec3 resonanceColor(std::uint32_t family) noexcept {
    switch (family) {
    case 1U: return {1.00, 0.56, 0.14}; // metal
    case 2U: return {0.72, 0.62, 0.48}; // mineral
    case 3U: return {0.86, 0.42, 0.12}; // wood
    case 4U: return {1.00, 0.18, 0.12}; // rubber
    case 5U: return {0.12, 1.00, 0.32}; // goo
    case 6U: return {0.98, 0.62, 0.92}; // plush
    case 7U: return {1.00, 0.24, 0.72}; // latex
    case 8U: return {0.20, 0.90, 1.00}; // foam
    default: return {0.22, 0.72, 1.00};
    }
}

[[nodiscard]] Vec3 kineticHeatColor(double energyJoules) noexcept {
    const double normalized = clamp(
        std::log10(1.0 + std::max(0.0, energyJoules)) / 4.0, 0.0, 1.0);
    const auto mix = [](const Vec3& from, const Vec3& to, double amount) {
        return from * (1.0 - amount) + to * amount;
    };
    if (normalized < 0.34) {
        return mix({0.025, 0.075, 0.30}, {0.02, 0.92, 1.00},
                   normalized / 0.34);
    }
    if (normalized < 0.68) {
        return mix({0.02, 0.92, 1.00}, {1.00, 0.72, 0.04},
                   (normalized - 0.34) / 0.34);
    }
    return mix({1.00, 0.72, 0.04}, {1.00, 0.08, 0.04},
               (normalized - 0.68) / 0.32);
}

[[nodiscard]] std::pair<std::vector<Vertex>, std::vector<std::uint16_t>>
cubeGeometry() {
    std::vector<Vertex> vertices;
    std::vector<std::uint16_t> indices;
    vertices.reserve(24U);
    indices.reserve(36U);
    constexpr std::array<std::array<float, 3>, 6> normals{{
        {{1.0F, 0.0F, 0.0F}}, {{-1.0F, 0.0F, 0.0F}},
        {{0.0F, 1.0F, 0.0F}}, {{0.0F, -1.0F, 0.0F}},
        {{0.0F, 0.0F, 1.0F}}, {{0.0F, 0.0F, -1.0F}},
    }};
    constexpr std::array<std::array<std::array<float, 3>, 4>, 6> faces{{
        {{{{.5F,-.5F,-.5F}},{{.5F,.5F,-.5F}},{{.5F,.5F,.5F}},{{.5F,-.5F,.5F}}}},
        {{{{-.5F,-.5F,.5F}},{{-.5F,.5F,.5F}},{{-.5F,.5F,-.5F}},{{-.5F,-.5F,-.5F}}}},
        {{{{-.5F,.5F,-.5F}},{{-.5F,.5F,.5F}},{{.5F,.5F,.5F}},{{.5F,.5F,-.5F}}}},
        {{{{-.5F,-.5F,.5F}},{{-.5F,-.5F,-.5F}},{{.5F,-.5F,-.5F}},{{.5F,-.5F,.5F}}}},
        {{{{-.5F,-.5F,.5F}},{{.5F,-.5F,.5F}},{{.5F,.5F,.5F}},{{-.5F,.5F,.5F}}}},
        {{{{.5F,-.5F,-.5F}},{{-.5F,-.5F,-.5F}},{{-.5F,.5F,-.5F}},{{.5F,.5F,-.5F}}}},
    }};
    constexpr std::array<std::uint16_t, 6> triangleOrder{{0U, 1U, 2U, 0U, 2U, 3U}};
    constexpr std::array<std::array<float, 2>, 4> textureCoordinates{{
        {{0.0F, 0.0F}}, {{0.0F, 1.0F}},
        {{1.0F, 1.0F}}, {{1.0F, 0.0F}},
    }};
    for (std::size_t face = 0U; face < faces.size(); ++face) {
        const auto base = static_cast<std::uint16_t>(vertices.size());
        const Vec3 tangent = Vec3{
            static_cast<Real>(faces[face][2][0] - faces[face][1][0]),
            static_cast<Real>(faces[face][2][1] - faces[face][1][1]),
            static_cast<Real>(faces[face][2][2] - faces[face][1][2]),
        }.normalized({1.0, 0.0, 0.0});
        const Vec3 bitangent = Vec3{
            static_cast<Real>(faces[face][1][0] - faces[face][0][0]),
            static_cast<Real>(faces[face][1][1] - faces[face][0][1]),
            static_cast<Real>(faces[face][1][2] - faces[face][0][2]),
        }.normalized({0.0, 1.0, 0.0});
        const Vec3 normal{
            static_cast<Real>(normals[face][0]),
            static_cast<Real>(normals[face][1]),
            static_cast<Real>(normals[face][2]),
        };
        const float handedness = normal.cross(tangent).dot(bitangent) < 0.0
            ? -1.0F : 1.0F;
        for (std::size_t corner = 0U; corner < faces[face].size(); ++corner) {
            const auto& point = faces[face][corner];
            vertices.push_back({
                {point[0], point[1], point[2]},
                {normals[face][0], normals[face][1], normals[face][2]},
                {gpu(tangent.x), gpu(tangent.y), gpu(tangent.z), handedness},
                {textureCoordinates[corner][0], textureCoordinates[corner][1]},
            });
        }
        for (std::uint16_t index : triangleOrder) {
            indices.push_back(static_cast<std::uint16_t>(base + index));
        }
    }
    return {std::move(vertices), std::move(indices)};
}

[[nodiscard]] std::pair<std::vector<Vertex>, std::vector<std::uint16_t>>
sphereGeometry(std::uint32_t slices, std::uint32_t stacks) {
    std::vector<Vertex> vertices;
    std::vector<std::uint16_t> indices;
    vertices.reserve(static_cast<std::size_t>((slices + 1U) * (stacks + 1U)));
    indices.reserve(static_cast<std::size_t>(slices * stacks * 6U));
    for (std::uint32_t stack = 0U; stack <= stacks; ++stack) {
        const float latitude = -std::numbers::pi_v<float> * 0.5F
            + std::numbers::pi_v<float> * static_cast<float>(stack)
              / static_cast<float>(stacks);
        const float y = std::sin(latitude);
        const float ring = std::cos(latitude);
        for (std::uint32_t slice = 0U; slice <= slices; ++slice) {
            const float longitude = 2.0F * std::numbers::pi_v<float>
                * static_cast<float>(slice) / static_cast<float>(slices);
            const float x = ring * std::sin(longitude);
            const float z = ring * std::cos(longitude);
            vertices.push_back({
                {x, y, z},
                {x, y, z},
                {std::cos(longitude), 0.0F, -std::sin(longitude), 1.0F},
                {longitude, latitude + std::numbers::pi_v<float> * 0.5F},
            });
        }
    }
    for (std::uint32_t stack = 0U; stack < stacks; ++stack) {
        for (std::uint32_t slice = 0U; slice < slices; ++slice) {
            const std::uint32_t firstValue = stack * (slices + 1U) + slice;
            const std::uint32_t secondValue = firstValue + slices + 1U;
            const auto first = static_cast<std::uint16_t>(firstValue);
            const auto second = static_cast<std::uint16_t>(secondValue);
            // The generated normals point outwards. Match the index winding to
            // those normals so the compensated Vulkan projection can use the
            // same stable CCW front face as the authored cube.
            indices.insert(indices.end(), {
                first, static_cast<std::uint16_t>(first + 1U), second,
                static_cast<std::uint16_t>(first + 1U),
                static_cast<std::uint16_t>(second + 1U), second,
            });
        }
    }
    return {std::move(vertices), std::move(indices)};
}

struct PpmAtlas {
    std::filesystem::path path;
    std::vector<std::uint8_t> rgb;
};

struct TextureUpload {
    std::vector<std::uint8_t> pixels;
    std::vector<VkBufferImageCopy> regions;
};

[[nodiscard]] bool ppmWhitespace(std::uint8_t value) noexcept {
    return value == static_cast<std::uint8_t>(' ')
        || value == static_cast<std::uint8_t>('\t')
        || value == static_cast<std::uint8_t>('\r')
        || value == static_cast<std::uint8_t>('\n')
        || value == static_cast<std::uint8_t>('\f')
        || value == static_cast<std::uint8_t>('\v');
}

[[nodiscard]] std::string_view ppmToken(const std::vector<std::uint8_t>& bytes,
                                        std::size_t& cursor,
                                        const std::filesystem::path& path) {
    while (cursor < bytes.size()) {
        while (cursor < bytes.size() && ppmWhitespace(bytes[cursor])) {
            ++cursor;
        }
        if (cursor >= bytes.size()
            || bytes[cursor] != static_cast<std::uint8_t>('#')) {
            break;
        }
        while (cursor < bytes.size()
               && bytes[cursor] != static_cast<std::uint8_t>('\n')) {
            ++cursor;
        }
    }
    const std::size_t begin = cursor;
    while (cursor < bytes.size() && !ppmWhitespace(bytes[cursor])
           && bytes[cursor] != static_cast<std::uint8_t>('#')) {
        ++cursor;
    }
    if (begin == cursor) {
        throw std::runtime_error("Malformed PPM header in texture atlas: "
                                 + path.string());
    }
    return {reinterpret_cast<const char*>(bytes.data() + begin), cursor - begin};
}

[[nodiscard]] std::uint32_t ppmUnsigned(std::string_view token,
                                        const std::filesystem::path& path,
                                        std::string_view field) {
    std::uint32_t value = 0U;
    const char* const end = token.data() + token.size();
    const auto result = std::from_chars(token.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::runtime_error("Invalid " + std::string(field)
            + " in PPM texture atlas: " + path.string());
    }
    return value;
}

[[nodiscard]] PpmAtlas readMaterialAtlas(std::string_view fileName) {
    namespace fs = std::filesystem;
    std::vector<fs::path> directories;
    std::error_code pathError;
    const fs::path executable = fs::read_symlink("/proc/self/exe", pathError);
    if (!pathError && !executable.empty()) {
        const fs::path binaryDirectory = executable.parent_path();
        directories.push_back(binaryDirectory / "textures");
        directories.push_back(binaryDirectory / ".." / "share"
            / "newtons_echo_chamber" / "textures");
    }
    directories.emplace_back(NEC_TEXTURE_ASSET_DIR);
    directories.emplace_back(NEC_INSTALL_TEXTURE_DIR);

    fs::path selected;
    std::ifstream stream;
    for (const fs::path& directory : directories) {
        const fs::path candidate = directory / fileName;
        stream.open(candidate, std::ios::binary | std::ios::ate);
        if (stream) {
            selected = candidate;
            break;
        }
        stream.clear();
    }
    if (!stream.is_open()) {
        std::string attempted;
        for (const fs::path& directory : directories) {
            if (!attempted.empty()) {
                attempted += ", ";
            }
            attempted += (directory / fileName).string();
        }
        throw std::runtime_error("Cannot find material texture atlas "
            + std::string(fileName) + "; searched: " + attempted);
    }

    const std::streampos end = stream.tellg();
    if (end <= std::streampos{0}) {
        throw std::runtime_error("Material texture atlas is empty: "
                                 + selected.string());
    }
    const auto byteCount = static_cast<std::size_t>(end);
    std::vector<std::uint8_t> fileBytes(byteCount);
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char*>(fileBytes.data()),
                static_cast<std::streamsize>(byteCount));
    if (!stream) {
        throw std::runtime_error("Cannot read material texture atlas: "
                                 + selected.string());
    }

    std::size_t cursor = 0U;
    if (ppmToken(fileBytes, cursor, selected) != "P6") {
        throw std::runtime_error("Material texture atlas must be binary P6 PPM: "
                                 + selected.string());
    }
    const std::uint32_t width = ppmUnsigned(
        ppmToken(fileBytes, cursor, selected), selected, "width");
    const std::uint32_t height = ppmUnsigned(
        ppmToken(fileBytes, cursor, selected), selected, "height");
    const std::uint32_t maximum = ppmUnsigned(
        ppmToken(fileBytes, cursor, selected), selected, "maximum sample");
    if (width != kMaterialAtlasExtent || height != kMaterialAtlasExtent) {
        throw std::runtime_error("Material texture atlas must be exactly 1024x1024: "
                                 + selected.string());
    }
    if (maximum != 255U) {
        throw std::runtime_error("Material texture atlas must use 8-bit PPM samples: "
                                 + selected.string());
    }
    if (cursor >= fileBytes.size() || !ppmWhitespace(fileBytes[cursor])) {
        throw std::runtime_error("PPM texture atlas header has no raster delimiter: "
                                 + selected.string());
    }
    if (fileBytes[cursor] == static_cast<std::uint8_t>('\r')
        && cursor + 1U < fileBytes.size()
        && fileBytes[cursor + 1U] == static_cast<std::uint8_t>('\n')) {
        cursor += 2U;
    } else {
        ++cursor;
    }

    constexpr std::size_t expected =
        static_cast<std::size_t>(kMaterialAtlasExtent)
        * static_cast<std::size_t>(kMaterialAtlasExtent) * 3U;
    if (cursor > fileBytes.size() || fileBytes.size() - cursor < expected) {
        throw std::runtime_error("PPM texture atlas raster is truncated: "
                                 + selected.string());
    }
    for (std::size_t trailing = cursor + expected;
         trailing < fileBytes.size(); ++trailing) {
        if (!ppmWhitespace(fileBytes[trailing])) {
            throw std::runtime_error("PPM texture atlas has unexpected trailing data: "
                                     + selected.string());
        }
    }
    std::vector<std::uint8_t> rgb(expected);
    std::memcpy(rgb.data(), fileBytes.data() + cursor, expected);
    return {std::move(selected), std::move(rgb)};
}

[[nodiscard]] std::uint8_t linearToSrgbByte(float linear) noexcept {
    const float bounded = clamp(linear, 0.0F, 1.0F);
    const float encoded = bounded <= 0.0031308F
        ? bounded * 12.92F
        : 1.055F * std::pow(bounded, 1.0F / 2.4F) - 0.055F;
    return static_cast<std::uint8_t>(clamp(
        static_cast<int>(std::lround(encoded * 255.0F)), 0, 255));
}

[[nodiscard]] const std::array<float, 256>& srgbLinearTable() noexcept {
    static const std::array<float, 256> table = [] {
        std::array<float, 256> values{};
        for (std::size_t index = 0U; index < values.size(); ++index) {
            const float encoded = static_cast<float>(index) / 255.0F;
            values[index] = encoded <= 0.04045F
                ? encoded / 12.92F
                : std::pow((encoded + 0.055F) / 1.055F, 2.4F);
        }
        return values;
    }();
    return table;
}

[[nodiscard]] std::vector<std::uint8_t>
downsampleRgba(const std::vector<std::uint8_t>& source,
               std::uint32_t sourceExtent, bool normalMap) {
    const std::uint32_t destinationExtent = std::max(1U, sourceExtent / 2U);
    std::vector<std::uint8_t> destination(
        static_cast<std::size_t>(destinationExtent)
        * static_cast<std::size_t>(destinationExtent) * 4U);
    const auto& linearTable = srgbLinearTable();
    for (std::uint32_t y = 0U; y < destinationExtent; ++y) {
        for (std::uint32_t x = 0U; x < destinationExtent; ++x) {
            const std::size_t destinationIndex =
                (static_cast<std::size_t>(y) * destinationExtent + x) * 4U;
            std::array<std::size_t, 4> sources{};
            std::size_t sample = 0U;
            for (std::uint32_t offsetY = 0U; offsetY < 2U; ++offsetY) {
                for (std::uint32_t offsetX = 0U; offsetX < 2U; ++offsetX) {
                    const std::uint32_t sourceX = std::min(
                        x * 2U + offsetX, sourceExtent - 1U);
                    const std::uint32_t sourceY = std::min(
                        y * 2U + offsetY, sourceExtent - 1U);
                    sources[sample++] =
                        (static_cast<std::size_t>(sourceY) * sourceExtent
                         + sourceX) * 4U;
                }
            }
            if (normalMap) {
                std::array<float, 3> summed{};
                for (std::size_t sourceIndex : sources) {
                    for (std::size_t component = 0U; component < 3U; ++component) {
                        summed[component] +=
                            static_cast<float>(source[sourceIndex + component])
                            * (2.0F / 255.0F) - 1.0F;
                    }
                }
                const float length = std::sqrt(
                    summed[0] * summed[0] + summed[1] * summed[1]
                    + summed[2] * summed[2]);
                if (length > 0.00001F) {
                    for (float& component : summed) {
                        component /= length;
                    }
                } else {
                    summed = {0.0F, 0.0F, 1.0F};
                }
                for (std::size_t component = 0U; component < 3U; ++component) {
                    destination[destinationIndex + component] =
                        static_cast<std::uint8_t>(clamp(
                            static_cast<int>(std::lround(
                                (summed[component] * 0.5F + 0.5F) * 255.0F)),
                            0, 255));
                }
            } else {
                for (std::size_t component = 0U; component < 3U; ++component) {
                    float average = 0.0F;
                    for (std::size_t sourceIndex : sources) {
                        average += linearTable[source[sourceIndex + component]];
                    }
                    destination[destinationIndex + component] =
                        linearToSrgbByte(average * 0.25F);
                }
            }
            unsigned alpha = 0U;
            for (std::size_t sourceIndex : sources) {
                alpha += source[sourceIndex + 3U];
            }
            destination[destinationIndex + 3U] =
                static_cast<std::uint8_t>((alpha + 2U) / 4U);
        }
    }
    return destination;
}

[[nodiscard]] TextureUpload buildMaterialMipChain(const PpmAtlas& atlas,
                                                   bool normalMap) {
    TextureUpload upload;
    constexpr std::size_t bytesPerLayer =
        (static_cast<std::size_t>(kMaterialLayerExtent)
         * static_cast<std::size_t>(kMaterialLayerExtent) * 4U * 4U) / 3U + 4U;
    upload.pixels.reserve(bytesPerLayer * kMaterialLayerCount);
    upload.regions.reserve(kMaterialLayerCount * kMaterialMipLevels);
    for (std::uint32_t layer = 0U; layer < kMaterialLayerCount; ++layer) {
        std::vector<std::uint8_t> current(
            static_cast<std::size_t>(kMaterialLayerExtent)
            * static_cast<std::size_t>(kMaterialLayerExtent) * 4U);
        const std::uint32_t atlasX = (layer % kMaterialAtlasColumns)
                                   * kMaterialLayerExtent;
        const std::uint32_t atlasY = (layer / kMaterialAtlasColumns)
                                   * kMaterialLayerExtent;
        for (std::uint32_t y = 0U; y < kMaterialLayerExtent; ++y) {
            for (std::uint32_t x = 0U; x < kMaterialLayerExtent; ++x) {
                const std::size_t sourceIndex =
                    (static_cast<std::size_t>(atlasY + y)
                     * kMaterialAtlasExtent + atlasX + x) * 3U;
                const std::size_t destinationIndex =
                    (static_cast<std::size_t>(y) * kMaterialLayerExtent + x) * 4U;
                current[destinationIndex] = atlas.rgb[sourceIndex];
                current[destinationIndex + 1U] = atlas.rgb[sourceIndex + 1U];
                current[destinationIndex + 2U] = atlas.rgb[sourceIndex + 2U];
                current[destinationIndex + 3U] = 255U;
            }
        }

        std::uint32_t extent = kMaterialLayerExtent;
        for (std::uint32_t mip = 0U; mip < kMaterialMipLevels; ++mip) {
            VkBufferImageCopy region{};
            region.bufferOffset = static_cast<VkDeviceSize>(upload.pixels.size());
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = mip;
            region.imageSubresource.baseArrayLayer = layer;
            region.imageSubresource.layerCount = 1U;
            region.imageExtent = {extent, extent, 1U};
            upload.regions.push_back(region);
            upload.pixels.insert(upload.pixels.end(), current.begin(), current.end());
            if (mip + 1U < kMaterialMipLevels) {
                current = downsampleRgba(current, extent, normalMap);
                extent = std::max(1U, extent / 2U);
            }
        }
    }
    return upload;
}

[[nodiscard]] std::vector<std::uint32_t> readShader(std::string_view fileName) {
    namespace fs = std::filesystem;
    std::vector<fs::path> directories;
    std::error_code pathError;
    const fs::path executable = fs::read_symlink("/proc/self/exe", pathError);
    if (!pathError && !executable.empty()) {
        const fs::path binaryDirectory = executable.parent_path();
        directories.push_back(binaryDirectory / "shaders");
        directories.push_back(
            binaryDirectory / ".." / "share" / "newtons_echo_chamber" / "shaders");
    }
    directories.emplace_back(NEC_SHADER_BINARY_DIR);
    directories.emplace_back(NEC_INSTALL_SHADER_DIR);
    std::ifstream stream;
    std::string path;
    for (const fs::path& directory : directories) {
        const fs::path candidate = directory / fileName;
        stream.open(candidate, std::ios::binary | std::ios::ate);
        if (stream) {
            path = candidate.string();
            break;
        }
        stream.clear();
    }
    if (!stream.is_open()) {
        throw std::runtime_error("Cannot open SPIR-V shader in build or install directory: "
                                 + std::string(fileName));
    }
    const std::streampos end = stream.tellg();
    if (end <= std::streampos{0}) {
        throw std::runtime_error("SPIR-V shader is empty: " + path);
    }
    const auto byteCount = static_cast<std::size_t>(end);
    if ((byteCount % sizeof(std::uint32_t)) != 0U) {
        throw std::runtime_error("SPIR-V shader has invalid byte size: " + path);
    }
    std::vector<std::uint32_t> words(byteCount / sizeof(std::uint32_t));
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char*>(words.data()),
                static_cast<std::streamsize>(byteCount));
    if (!stream) {
        throw std::runtime_error("Cannot read SPIR-V shader: " + path);
    }
    return words;
}

VKAPI_ATTR VkBool32 VKAPI_CALL validationCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT*, void* userData) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0U
        && userData != nullptr) {
        auto* count = static_cast<std::atomic<std::uint32_t>*>(userData);
        count->fetch_add(1U, std::memory_order_relaxed);
    }
    return VK_FALSE;
}

struct Buffer {
    VkBuffer handle{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize size{};
    VkDeviceSize allocationSize{};
    void* mapped{};
    bool coherent{};
};

struct ImageResource {
    VkImage image{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
    bool initialized{};
};

struct TextureArray {
    VkImage image{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
};

struct ShadowMap {
    VkImage image{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
};

struct Mesh {
    Buffer vertices{};
    Buffer indices{};
    std::uint32_t indexCount{};
};

struct FrameContext {
    VkCommandPool commandPool{VK_NULL_HANDLE};
    VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
    VkFence fence{VK_NULL_HANDLE};
    VkSemaphore imageAvailable{VK_NULL_HANDLE};
    VkSemaphore renderFinished{VK_NULL_HANDLE};
    Buffer uniform{};
    Buffer instances{};
    Buffer impacts{};
    Buffer hudVertices{};
    VkDescriptorSet graphicsDescriptor{VK_NULL_HANDLE};
    VkDescriptorSet computeDescriptor{VK_NULL_HANDLE};
    VkQueryPool timestampQueries{VK_NULL_HANDLE};
    ShadowMap shadowMap{};
    std::uint32_t impactCount{};
    std::uint32_t impactEmissionCount{};
    std::uint32_t impactFirstParticle{};
    std::uint32_t impactEndParticle{};
    std::uint32_t hudVertexCount{};
    bool timestampPending{};
    bool timestampComputeActive{};
    bool timestampShadowActive{};
};

struct DeviceCandidate {
    VkPhysicalDevice device{VK_NULL_HANDLE};
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceDriverProperties driver = vkInitialize<VkPhysicalDeviceDriverProperties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
    VkPhysicalDeviceMemoryProperties memory{};
    std::vector<VkExtensionProperties> extensions;
    std::uint32_t queueFamily{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t timestampValidBits{};
    bool core13{};
    bool timeline{};
    bool memoryBudget{};
    bool software{};
    std::uint64_t score{};
};

} // namespace

struct Renderer::Impl {
    const Platform* platform{};
    VkInstance instance{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT debugMessenger{VK_NULL_HANDLE};
    VkSurfaceKHR surface{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue queue{VK_NULL_HANDLE};
    std::uint32_t queueFamily{};

    VkSwapchainKHR swapchain{VK_NULL_HANDLE};
    VkFormat colorFormat{VK_FORMAT_UNDEFINED};
    VkColorSpaceKHR colorSpace{VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    VkFormat depthFormat{VK_FORMAT_UNDEFINED};
    VkExtent2D extent{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainViews;
    std::vector<VkImageLayout> swapchainLayouts;
    std::vector<ImageResource> depthImages;
    std::vector<VkFence> imagesInFlight;

    std::array<FrameContext, kFrameCount> frames{};
    std::uint32_t frameIndex{};

    VkDescriptorSetLayout graphicsSetLayout{VK_NULL_HANDLE};
    VkDescriptorSetLayout computeSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool descriptorPool{VK_NULL_HANDLE};
    VkPipelineLayout graphicsPipelineLayout{VK_NULL_HANDLE};
    VkPipelineLayout computePipelineLayout{VK_NULL_HANDLE};
    VkPipeline scenePipeline{VK_NULL_HANDLE};
    VkPipeline resonancePipeline{VK_NULL_HANDLE};
    VkPipeline particlePipeline{VK_NULL_HANDLE};
    VkPipeline hudPipeline{VK_NULL_HANDLE};
    VkPipeline shadowPipeline{VK_NULL_HANDLE};
    VkPipeline computePipeline{VK_NULL_HANDLE};
    VkSampler shadowSampler{VK_NULL_HANDLE};
    VkSampler materialSampler{VK_NULL_HANDLE};
    VkFormat shadowFormat{VK_FORMAT_UNDEFINED};

    TextureArray materialAlbedo{};
    TextureArray materialNormal{};

    Mesh cube{};
    Mesh sphereNear{};
    Mesh sphereMedium{};
    Mesh sphereFar{};
    Buffer particles{};
    std::uint32_t particleCapacity{};
    std::uint32_t activeParticles{};
    std::uint32_t frameSerial{};
    std::uint32_t impactParticleCursor{};
    std::uint64_t simulatedFrames{};
    double gpuTime{};

    PFN_vkCmdBeginRendering beginRendering{};
    PFN_vkCmdEndRendering endRendering{};
    PFN_vkCmdPipelineBarrier2 pipelineBarrier2{};
    PFN_vkQueueSubmit2 queueSubmit2{};

    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    std::uint32_t localHeapIndex{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t loaderApiVersion{};
    std::string deviceName;
    std::string driverName;
    std::atomic<std::uint32_t> validationErrors{0U};
    std::uint64_t renderedFrames{};
    std::uint32_t timestampValidBits{};
    double timestampPeriodNanoseconds{};
    double gpuComputeMillisecondsSum{};
    double gpuShadowMillisecondsSum{};
    double gpuGraphicsMillisecondsSum{};
    double gpuTotalMillisecondsSum{};
    std::uint64_t timedFrames{};
    std::uint64_t shadowedFrames{};
    bool validationRequested{};
    bool validationEnabled{};
    bool debugUtilsEnabled{};
    bool core13{};
    bool timeline{};
    bool memoryBudget{};
    bool software{};
    bool ready{};
    bool lost{};
    bool maximumGpu{};
    bool gpuPhysicsEnabled{true};
    bool overheadLightsEnabled{true};
    bool flashlightEnabled{};
    bool flashlightShadowsEnabled{true};
    bool materialTexturesEnabled{true};
    bool normalMappingEnabled{true};
    bool shadowAvailable{};
    bool shadowLinearFiltering{};
    bool timestampsSupported{};
    int quality{};
    RendererHudState hudState{};
    int probeBody{-1};
    double lastImpactSimulationTime{-1.0};
    double lastResonanceSimulationTime{-1.0};
    std::array<ResonancePulse, kResonancePulseCapacity> resonancePulses{};
    std::size_t resonancePulseCursor{};
    double lastEchoPulseSimulationTime{-1.0};
    std::uint64_t lastEchoPulseSerial{};
    std::array<EchoPulseVisual, kEchoPulseVisualCapacity> echoPulseVisuals{};
    std::size_t echoPulseVisualCursor{};
    std::string error;

    std::vector<Instance> cubeInstances;
    std::vector<Instance> sphereNearInstances;
    std::vector<Instance> sphereMediumInstances;
    std::vector<Instance> sphereFarInstances;
    std::vector<Instance> resonanceInstances;
    std::vector<Instance> shadowCubeInstances;
    std::vector<Instance> shadowSphereNearInstances;
    std::vector<Instance> shadowSphereMediumInstances;
    std::vector<Instance> shadowSphereFarInstances;
    std::vector<HudVertex> hudVerticesCpu;

    void createInstance(bool enableValidation);
    void createSurface();
    void choosePhysicalDevice(bool allowSoftwareDevice);
    void createDevice();
    void loadDeviceCommands();
    void createFrameContexts();
    void chooseShadowFormat();
    void createShadowMap(ShadowMap& target);
    void destroyShadowMap(ShadowMap& target) noexcept;
    void createShadowSampler();
    void createTextureArray(TextureArray& target, VkFormat format);
    void destroyTextureArray(TextureArray& target) noexcept;
    void createMaterialTextures();
    void createMaterialSampler();
    void createDescriptors();
    void createMeshes();
    void createParticles(std::uint32_t seed);
    void createComputePipeline();
    void createShadowPipeline();
    [[nodiscard]] bool createSwapchain();
    void createGraphicsPipelines();
    void collectFrameTimestamps(FrameContext& frame) noexcept;
    void collectAllFrameTimestamps() noexcept;

    [[nodiscard]] std::optional<std::pair<std::uint32_t, bool>>
    findMemoryType(std::uint32_t typeBits, VkMemoryPropertyFlags required,
                   VkMemoryPropertyFlags preferred) const noexcept;
    void createBuffer(Buffer& buffer, VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags preferred =
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void destroyBuffer(Buffer& buffer) noexcept;
    void writeBuffer(Buffer& buffer, const void* source, std::size_t bytes,
                     VkDeviceSize offset = 0U);
    void createDepthImage(ImageResource& target);
    void destroyDepthImage(ImageResource& target) noexcept;
    [[nodiscard]] VkShaderModule createShaderModule(std::string_view name) const;

    void destroyGraphicsPipelines() noexcept;
    void destroySwapchain() noexcept;
    void cleanup() noexcept;
    [[nodiscard]] bool recreateSwapchain();
    [[nodiscard]] bool acquireImage(FrameContext& frame, std::uint32_t& imageIndex,
                                    bool& suboptimal);
    void updateScene(const PhysicsWorld& world, float alpha, FrameContext& frame);
    void appendInstance(std::vector<Instance>& destination, const Vec3& position,
                        const Quat& orientation, const Vec3& scale,
                        const Vec3& color, float material = 0.0F,
                        float emissive = 0.0F, float alpha = 1.0F);
    void record(FrameContext& frame, std::uint32_t imageIndex,
                const PhysicsWorld& world, float frameDeltaSeconds);
    [[nodiscard]] bool draw(const PhysicsWorld& world, float alpha,
                            float frameDeltaSeconds);
    [[nodiscard]] VulkanStats stats() const noexcept;
};

void Renderer::Impl::createInstance(bool enableValidation) {
    auto enumerateVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    loaderApiVersion = VK_API_VERSION_1_0;
    if (enumerateVersion != nullptr) {
        requireVk(enumerateVersion(&loaderApiVersion), "vkEnumerateInstanceVersion");
    }
    if (loaderApiVersion < VK_API_VERSION_1_3) {
        throw std::runtime_error(
            "Vulkan 1.3 loader is required; installed loader reports "
            + std::to_string(VK_API_VERSION_MAJOR(loaderApiVersion)) + "."
            + std::to_string(VK_API_VERSION_MINOR(loaderApiVersion)));
    }

    const auto availableExtensions = instanceExtensions();
    for (const char* required : {VK_KHR_SURFACE_EXTENSION_NAME,
                                 VK_KHR_XLIB_SURFACE_EXTENSION_NAME}) {
        if (!hasName(availableExtensions, required)) {
            throw std::runtime_error(std::string("Missing Vulkan instance extension: ")
                                     + required);
        }
    }

    validationRequested = enableValidation;
    std::vector<const char*> extensions{
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
    };
    std::vector<const char*> layers;
    bool synchronizationValidation = false;
    if (enableValidation) {
        const auto availableLayers = instanceLayers();
        constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
        validationEnabled = hasName(availableLayers, validationLayer);
        debugUtilsEnabled = validationEnabled
                         && hasName(availableExtensions,
                                    VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (validationEnabled) {
            layers.push_back(validationLayer);
        }
        if (debugUtilsEnabled) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        synchronizationValidation = validationEnabled
            && hasName(availableExtensions,
                       VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
        if (synchronizationValidation) {
            extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
        }
    }

    const VkApplicationInfo application{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "Newton's Echo Chamber",
        VK_MAKE_API_VERSION(0U, 1U, 0U, 0U),
        "NEC Vulkan Renderer",
        VK_MAKE_API_VERSION(0U, 1U, 0U, 0U),
        VK_API_VERSION_1_3,
    };
    VkDebugUtilsMessengerCreateInfoEXT debugCreate = vkInitialize<VkDebugUtilsMessengerCreateInfoEXT>(VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT);
    debugCreate.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreate.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreate.pfnUserCallback = validationCallback;
    debugCreate.pUserData = &validationErrors;

    constexpr VkValidationFeatureEnableEXT synchronizationFeature =
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT validationFeatures =
        vkInitialize<VkValidationFeaturesEXT>(
            VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT);
    validationFeatures.pNext = debugUtilsEnabled ? &debugCreate : nullptr;
    validationFeatures.enabledValidationFeatureCount = 1U;
    validationFeatures.pEnabledValidationFeatures = &synchronizationFeature;

    VkInstanceCreateInfo createInfo = vkInitialize<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    createInfo.pNext = synchronizationValidation
        ? static_cast<const void*>(&validationFeatures)
        : debugUtilsEnabled ? static_cast<const void*>(&debugCreate) : nullptr;
    createInfo.pApplicationInfo = &application;
    createInfo.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    requireVk(vkCreateInstance(&createInfo, nullptr, &instance), "vkCreateInstance");

    if (debugUtilsEnabled) {
        auto createDebug = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (createDebug != nullptr) {
            requireVk(createDebug(instance, &debugCreate, nullptr, &debugMessenger),
                      "vkCreateDebugUtilsMessengerEXT");
        }
    }
}

void Renderer::Impl::createSurface() {
    if (platform == nullptr || !platform->valid()
        || platform->nativeDisplay() == nullptr
        || platform->nativeWindow() == 0UL) {
        throw std::runtime_error("A valid Xlib Platform window is required");
    }
    VkXlibSurfaceCreateInfoKHR createInfo = vkInitialize<VkXlibSurfaceCreateInfoKHR>(VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR);
    createInfo.dpy = static_cast<Display*>(platform->nativeDisplay());
    createInfo.window = static_cast<Window>(platform->nativeWindow());
    requireVk(vkCreateXlibSurfaceKHR(instance, &createInfo, nullptr, &surface),
              "vkCreateXlibSurfaceKHR");
}

void Renderer::Impl::choosePhysicalDevice(bool allowSoftwareDevice) {
    std::uint32_t count = 0U;
    requireVk(vkEnumeratePhysicalDevices(instance, &count, nullptr),
              "vkEnumeratePhysicalDevices(count)");
    if (count == 0U) {
        throw std::runtime_error("No Vulkan physical devices were found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    requireVk(vkEnumeratePhysicalDevices(instance, &count, devices.data()),
              "vkEnumeratePhysicalDevices");
    devices.resize(count);

    std::optional<DeviceCandidate> best;
    for (VkPhysicalDevice candidateDevice : devices) {
        DeviceCandidate candidate;
        candidate.device = candidateDevice;
        VkPhysicalDeviceProperties2 properties2 = vkInitialize<VkPhysicalDeviceProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
        properties2.pNext = &candidate.driver;
        vkGetPhysicalDeviceProperties2(candidateDevice, &properties2);
        candidate.properties = properties2.properties;
        vkGetPhysicalDeviceMemoryProperties(candidateDevice, &candidate.memory);
        candidate.software = isSoftwareDevice(candidate.properties);
        if (candidate.software && !allowSoftwareDevice) {
            continue;
        }
        if (candidate.properties.apiVersion < VK_API_VERSION_1_2) {
            continue;
        }
        candidate.extensions = deviceExtensions(candidateDevice);
        if (!hasName(candidate.extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            continue;
        }

        std::uint32_t queueCount = 0U;
        vkGetPhysicalDeviceQueueFamilyProperties(candidateDevice, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(
            candidateDevice, &queueCount, queues.data());
        for (std::uint32_t family = 0U; family < queueCount; ++family) {
            VkBool32 present = VK_FALSE;
            if (vkGetPhysicalDeviceSurfaceSupportKHR(
                    candidateDevice, family, surface, &present) != VK_SUCCESS) {
                continue;
            }
            const VkQueueFlags required = VK_QUEUE_GRAPHICS_BIT
                                        | VK_QUEUE_COMPUTE_BIT;
            if ((queues[family].queueFlags & required) == required
                && present == VK_TRUE && queues[family].queueCount > 0U) {
                candidate.queueFamily = family;
                candidate.timestampValidBits = queues[family].timestampValidBits;
                break;
            }
        }
        if (candidate.queueFamily == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }

        VkPhysicalDeviceVulkan12Features features12 = vkInitialize<VkPhysicalDeviceVulkan12Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
        VkPhysicalDeviceFeatures2 features2 = vkInitialize<VkPhysicalDeviceFeatures2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
        features2.pNext = &features12;
        const bool api13 = candidate.properties.apiVersion >= VK_API_VERSION_1_3;
        VkPhysicalDeviceVulkan13Features features13 = vkInitialize<VkPhysicalDeviceVulkan13Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRendering = vkInitialize<VkPhysicalDeviceDynamicRenderingFeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR);
        VkPhysicalDeviceSynchronization2FeaturesKHR synchronization2 = vkInitialize<VkPhysicalDeviceSynchronization2FeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR);
        VkPhysicalDeviceMaintenance4FeaturesKHR maintenance4 = vkInitialize<VkPhysicalDeviceMaintenance4FeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR);
        if (api13) {
            features12.pNext = &features13;
        } else {
            const bool extensionProfile =
                hasName(candidate.extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)
                && hasName(candidate.extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
                && hasName(candidate.extensions, VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
            if (!extensionProfile) {
                continue;
            }
            features12.pNext = &dynamicRendering;
            dynamicRendering.pNext = &synchronization2;
            synchronization2.pNext = &maintenance4;
        }
        vkGetPhysicalDeviceFeatures2(candidateDevice, &features2);
        const bool profileFeatures = api13
            ? features13.dynamicRendering == VK_TRUE
              && features13.synchronization2 == VK_TRUE
              && features13.maintenance4 == VK_TRUE
            : dynamicRendering.dynamicRendering == VK_TRUE
              && synchronization2.synchronization2 == VK_TRUE
              && maintenance4.maintenance4 == VK_TRUE;
        if (!profileFeatures) {
            continue;
        }
        candidate.core13 = api13;
        candidate.timeline = features12.timelineSemaphore == VK_TRUE;
        candidate.memoryBudget = hasName(candidate.extensions,
                                         VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

        std::uint64_t score = isV3dv(candidate.properties.deviceName)
            && !candidate.software ? 1'000'000ULL : 0ULL;
        switch (candidate.properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score += 50'000ULL; break;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score += 40'000ULL; break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score += 10'000ULL; break;
        case VK_PHYSICAL_DEVICE_TYPE_OTHER: score += 5'000ULL; break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: score += 1ULL; break;
        default: break;
        }
        score += static_cast<std::uint64_t>(
            VK_API_VERSION_MINOR(candidate.properties.apiVersion)) * 100ULL;
        candidate.score = score;
        if (!best || candidate.score > best->score) {
            best = std::move(candidate);
        }
    }

    if (!best) {
        throw std::runtime_error(
            "No acceptable graphics+compute+present Vulkan device supports the "
            "dynamic-rendering, synchronization2 and maintenance4 profile");
    }
    physicalDevice = best->device;
    properties = best->properties;
    memoryProperties = best->memory;
    queueFamily = best->queueFamily;
    timestampValidBits = best->timestampValidBits;
    timestampPeriodNanoseconds = static_cast<double>(properties.limits.timestampPeriod);
    timestampsSupported = timestampValidBits != 0U
        && timestampPeriodNanoseconds > 0.0
        && properties.limits.timestampComputeAndGraphics == VK_TRUE;
    core13 = best->core13;
    timeline = best->timeline;
    memoryBudget = best->memoryBudget;
    software = best->software;
    deviceName = properties.deviceName;
    driverName = best->driver.driverName;
    if (driverName.empty()) {
        driverName = "driver " + std::to_string(properties.driverVersion);
    }

    VkDeviceSize largest = 0U;
    for (std::uint32_t heap = 0U; heap < memoryProperties.memoryHeapCount; ++heap) {
        if ((memoryProperties.memoryHeaps[heap].flags
             & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0U
            && memoryProperties.memoryHeaps[heap].size > largest) {
            largest = memoryProperties.memoryHeaps[heap].size;
            localHeapIndex = heap;
        }
    }
}

void Renderer::Impl::createDevice() {
    constexpr float priority = 1.0F;
    VkDeviceQueueCreateInfo queueCreate = vkInitialize<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queueCreate.queueFamilyIndex = queueFamily;
    queueCreate.queueCount = 1U;
    queueCreate.pQueuePriorities = &priority;

    std::vector<const char*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (!core13) {
        extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
        extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
        extensions.push_back(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    }
    if (memoryBudget) {
        extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }

    VkPhysicalDeviceVulkan12Features features12 = vkInitialize<VkPhysicalDeviceVulkan12Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    features12.timelineSemaphore = timeline ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceVulkan13Features features13 = vkInitialize<VkPhysicalDeviceVulkan13Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRendering = vkInitialize<VkPhysicalDeviceDynamicRenderingFeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR);
    VkPhysicalDeviceSynchronization2FeaturesKHR synchronization2 = vkInitialize<VkPhysicalDeviceSynchronization2FeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR);
    VkPhysicalDeviceMaintenance4FeaturesKHR maintenance4 = vkInitialize<VkPhysicalDeviceMaintenance4FeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR);
    if (core13) {
        features12.pNext = &features13;
        features13.dynamicRendering = VK_TRUE;
        features13.synchronization2 = VK_TRUE;
        features13.maintenance4 = VK_TRUE;
    } else {
        features12.pNext = &dynamicRendering;
        dynamicRendering.dynamicRendering = VK_TRUE;
        dynamicRendering.pNext = &synchronization2;
        synchronization2.synchronization2 = VK_TRUE;
        synchronization2.pNext = &maintenance4;
        maintenance4.maintenance4 = VK_TRUE;
    }

    VkDeviceCreateInfo createInfo = vkInitialize<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    createInfo.pNext = &features12;
    createInfo.queueCreateInfoCount = 1U;
    createInfo.pQueueCreateInfos = &queueCreate;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    requireVk(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device),
              "vkCreateDevice");
    vkGetDeviceQueue(device, queueFamily, 0U, &queue);
}

void Renderer::Impl::loadDeviceCommands() {
    auto load = [this](const char* coreName, const char* extensionName) {
        PFN_vkVoidFunction command = vkGetDeviceProcAddr(device, coreName);
        if (command == nullptr) {
            command = vkGetDeviceProcAddr(device, extensionName);
        }
        return command;
    };
    beginRendering = reinterpret_cast<PFN_vkCmdBeginRendering>(
        load("vkCmdBeginRendering", "vkCmdBeginRenderingKHR"));
    endRendering = reinterpret_cast<PFN_vkCmdEndRendering>(
        load("vkCmdEndRendering", "vkCmdEndRenderingKHR"));
    pipelineBarrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(
        load("vkCmdPipelineBarrier2", "vkCmdPipelineBarrier2KHR"));
    queueSubmit2 = reinterpret_cast<PFN_vkQueueSubmit2>(
        load("vkQueueSubmit2", "vkQueueSubmit2KHR"));
    if (beginRendering == nullptr || endRendering == nullptr
        || pipelineBarrier2 == nullptr || queueSubmit2 == nullptr) {
        throw std::runtime_error(
            "The selected Vulkan device did not expose required 1.3/KHR commands");
    }
}

std::optional<std::pair<std::uint32_t, bool>> Renderer::Impl::findMemoryType(
    std::uint32_t typeBits, VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred) const noexcept {
    std::optional<std::pair<std::uint32_t, bool>> fallback;
    for (std::uint32_t index = 0U; index < memoryProperties.memoryTypeCount; ++index) {
        if ((typeBits & (1U << index)) == 0U) {
            continue;
        }
        const VkMemoryPropertyFlags flags = memoryProperties.memoryTypes[index].propertyFlags;
        if ((flags & required) != required) {
            continue;
        }
        const bool coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U;
        if ((flags & preferred) == preferred) {
            return std::pair<std::uint32_t, bool>{index, coherent};
        }
        if (!fallback) {
            fallback = std::pair<std::uint32_t, bool>{index, coherent};
        }
    }
    return fallback;
}

void Renderer::Impl::createBuffer(Buffer& buffer, VkDeviceSize size,
                                  VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags preferred) {
    buffer.size = size;
    VkBufferCreateInfo createInfo = vkInitialize<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    createInfo.size = size;
    createInfo.usage = usage;
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    requireVk(vkCreateBuffer(device, &createInfo, nullptr, &buffer.handle),
              "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer.handle, &requirements);
    const auto memoryType = findMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, preferred);
    if (!memoryType) {
        throw std::runtime_error("No host-visible Vulkan memory type is available");
    }
    VkMemoryAllocateInfo allocation = vkInitialize<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType->first;
    requireVk(vkAllocateMemory(device, &allocation, nullptr, &buffer.memory),
              "vkAllocateMemory(buffer)");
    buffer.allocationSize = requirements.size;
    buffer.coherent = memoryType->second;
    requireVk(vkBindBufferMemory(device, buffer.handle, buffer.memory, 0U),
              "vkBindBufferMemory");
    requireVk(vkMapMemory(device, buffer.memory, 0U, VK_WHOLE_SIZE, 0U,
                          &buffer.mapped),
              "vkMapMemory(buffer)");
}

void Renderer::Impl::destroyBuffer(Buffer& buffer) noexcept {
    if (device != VK_NULL_HANDLE && buffer.memory != VK_NULL_HANDLE
        && buffer.mapped != nullptr) {
        vkUnmapMemory(device, buffer.memory);
    }
    if (device != VK_NULL_HANDLE && buffer.handle != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer.handle, nullptr);
    }
    if (device != VK_NULL_HANDLE && buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer.memory, nullptr);
    }
    buffer = {};
}

void Renderer::Impl::writeBuffer(Buffer& buffer, const void* source,
                                 std::size_t bytes, VkDeviceSize offset) {
    if (buffer.mapped == nullptr || source == nullptr
        || offset > buffer.size
        || static_cast<VkDeviceSize>(bytes) > buffer.size - offset) {
        throw std::runtime_error("Vulkan host buffer write exceeded its allocation");
    }
    auto* destination = static_cast<std::byte*>(buffer.mapped)
                      + static_cast<std::size_t>(offset);
    std::memcpy(destination, source, bytes);
    if (!buffer.coherent) {
        // Every host buffer has a dedicated allocation and offset zero is
        // nonCoherentAtomSize aligned. Flushing the whole allocation also
        // avoids rounding beyond allocationSize on the final atom.
        VkMappedMemoryRange range = vkInitialize<VkMappedMemoryRange>(VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
        range.memory = buffer.memory;
        range.offset = 0U;
        range.size = VK_WHOLE_SIZE;
        requireVk(vkFlushMappedMemoryRanges(device, 1U, &range),
                  "vkFlushMappedMemoryRanges");
    }
}

void Renderer::Impl::chooseShadowFormat() {
    constexpr std::array<VkFormat, 3> candidates{{
        VK_FORMAT_D16_UNORM,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT,
    }};
    constexpr VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

    shadowFormat = VK_FORMAT_UNDEFINED;
    shadowLinearFiltering = false;
    // Prefer a hardware-filterable D16 target. If that exact combination is
    // unavailable, retain comparison sampling and fall back to four nearest
    // taps in the shader rather than disabling the renderer.
    for (bool requireLinear : {true, false}) {
        for (VkFormat format : candidates) {
            VkFormatProperties formatProperties{};
            vkGetPhysicalDeviceFormatProperties(
                physicalDevice, format, &formatProperties);
            const VkFormatFeatureFlags features =
                formatProperties.optimalTilingFeatures;
            if ((features & required) != required) {
                continue;
            }
            const bool linear = (features
                & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0U;
            if (requireLinear && !linear) {
                continue;
            }
            shadowFormat = format;
            shadowLinearFiltering = linear;
            return;
        }
    }
    throw std::runtime_error(
        "No sampled depth-attachment format is available for flashlight shadows");
}

void Renderer::Impl::createShadowMap(ShadowMap& target) {
    VkImageCreateInfo imageCreate = vkInitialize<VkImageCreateInfo>(
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    imageCreate.imageType = VK_IMAGE_TYPE_2D;
    imageCreate.format = shadowFormat;
    imageCreate.extent = {kShadowResolution, kShadowResolution, 1U};
    imageCreate.mipLevels = 1U;
    imageCreate.arrayLayers = 1U;
    imageCreate.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreate.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreate.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                      | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreate.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreate.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    requireVk(vkCreateImage(device, &imageCreate, nullptr, &target.image),
              "vkCreateImage(flashlight shadow)");

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, target.image, &requirements);
    const auto memoryType = findMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!memoryType) {
        throw std::runtime_error(
            "No device-local memory type for flashlight shadow map");
    }
    VkMemoryAllocateInfo allocation = vkInitialize<VkMemoryAllocateInfo>(
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType->first;
    requireVk(vkAllocateMemory(device, &allocation, nullptr, &target.memory),
              "vkAllocateMemory(flashlight shadow)");
    requireVk(vkBindImageMemory(device, target.image, target.memory, 0U),
              "vkBindImageMemory(flashlight shadow)");

    VkImageViewCreateInfo viewCreate = vkInitialize<VkImageViewCreateInfo>(
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
    viewCreate.image = target.image;
    viewCreate.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreate.format = shadowFormat;
    viewCreate.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewCreate.subresourceRange.levelCount = 1U;
    viewCreate.subresourceRange.layerCount = 1U;
    requireVk(vkCreateImageView(device, &viewCreate, nullptr, &target.view),
              "vkCreateImageView(flashlight shadow)");
}

void Renderer::Impl::destroyShadowMap(ShadowMap& target) noexcept {
    if (device != VK_NULL_HANDLE && target.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, target.view, nullptr);
    }
    if (device != VK_NULL_HANDLE && target.image != VK_NULL_HANDLE) {
        vkDestroyImage(device, target.image, nullptr);
    }
    if (device != VK_NULL_HANDLE && target.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, target.memory, nullptr);
    }
    target = {};
}

void Renderer::Impl::createShadowSampler() {
    VkSamplerCreateInfo samplerCreate = vkInitialize<VkSamplerCreateInfo>(
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);
    samplerCreate.magFilter = shadowLinearFiltering
        ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    samplerCreate.minFilter = samplerCreate.magFilter;
    samplerCreate.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreate.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreate.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreate.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreate.mipLodBias = 0.0F;
    samplerCreate.anisotropyEnable = VK_FALSE;
    samplerCreate.maxAnisotropy = 1.0F;
    samplerCreate.compareEnable = VK_TRUE;
    samplerCreate.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    samplerCreate.minLod = 0.0F;
    samplerCreate.maxLod = 0.0F;
    samplerCreate.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerCreate.unnormalizedCoordinates = VK_FALSE;
    requireVk(vkCreateSampler(device, &samplerCreate, nullptr, &shadowSampler),
              "vkCreateSampler(flashlight shadow)");
}

void Renderer::Impl::createTextureArray(TextureArray& target, VkFormat format) {
    VkFormatProperties formatProperties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
    constexpr VkFormatFeatureFlags requiredFeatures =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
        | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
        | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if ((formatProperties.optimalTilingFeatures & requiredFeatures)
        != requiredFeatures) {
        throw std::runtime_error(
            "Selected Vulkan device cannot linearly sample the required "
            "RGBA8 material texture-array format");
    }
    if (properties.limits.maxImageDimension2D < kMaterialLayerExtent
        || properties.limits.maxImageArrayLayers < kMaterialLayerCount) {
        throw std::runtime_error(
            "Selected Vulkan device cannot create the 256px, 16-layer "
            "material texture arrays");
    }

    VkImageCreateInfo imageCreate = vkInitialize<VkImageCreateInfo>(
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    imageCreate.imageType = VK_IMAGE_TYPE_2D;
    imageCreate.format = format;
    imageCreate.extent = {
        kMaterialLayerExtent, kMaterialLayerExtent, 1U};
    imageCreate.mipLevels = kMaterialMipLevels;
    imageCreate.arrayLayers = kMaterialLayerCount;
    imageCreate.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreate.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreate.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT
                      | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreate.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreate.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    requireVk(vkCreateImage(device, &imageCreate, nullptr, &target.image),
              "vkCreateImage(material texture array)");

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, target.image, &requirements);
    const auto memoryType = findMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!memoryType) {
        throw std::runtime_error(
            "No device-local memory type is available for material textures");
    }
    VkMemoryAllocateInfo allocation = vkInitialize<VkMemoryAllocateInfo>(
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType->first;
    requireVk(vkAllocateMemory(device, &allocation, nullptr, &target.memory),
              "vkAllocateMemory(material texture array)");
    requireVk(vkBindImageMemory(device, target.image, target.memory, 0U),
              "vkBindImageMemory(material texture array)");

    VkImageViewCreateInfo viewCreate = vkInitialize<VkImageViewCreateInfo>(
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
    viewCreate.image = target.image;
    viewCreate.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewCreate.format = format;
    viewCreate.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCreate.subresourceRange.levelCount = kMaterialMipLevels;
    viewCreate.subresourceRange.layerCount = kMaterialLayerCount;
    requireVk(vkCreateImageView(device, &viewCreate, nullptr, &target.view),
              "vkCreateImageView(material texture array)");
}

void Renderer::Impl::destroyTextureArray(TextureArray& target) noexcept {
    if (device != VK_NULL_HANDLE && target.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, target.view, nullptr);
    }
    if (device != VK_NULL_HANDLE && target.image != VK_NULL_HANDLE) {
        vkDestroyImage(device, target.image, nullptr);
    }
    if (device != VK_NULL_HANDLE && target.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, target.memory, nullptr);
    }
    target = {};
}

void Renderer::Impl::createMaterialTextures() {
    const PpmAtlas albedoAtlas = readMaterialAtlas("material_albedo.ppm");
    const PpmAtlas normalAtlas = readMaterialAtlas("material_normal.ppm");
    TextureUpload albedoUpload = buildMaterialMipChain(albedoAtlas, false);
    TextureUpload normalUpload = buildMaterialMipChain(normalAtlas, true);
    if (albedoUpload.pixels.empty() || normalUpload.pixels.empty()
        || albedoUpload.regions.size()
            != static_cast<std::size_t>(kMaterialLayerCount
                                        * kMaterialMipLevels)
        || normalUpload.regions.size() != albedoUpload.regions.size()) {
        throw std::runtime_error(
            "Material atlas conversion did not produce all texture layers/mips");
    }

    createTextureArray(materialAlbedo, VK_FORMAT_R8G8B8A8_SRGB);
    createTextureArray(materialNormal, VK_FORMAT_R8G8B8A8_UNORM);

    const VkDeviceSize normalOffset =
        static_cast<VkDeviceSize>(albedoUpload.pixels.size());
    const VkDeviceSize totalBytes = normalOffset
        + static_cast<VkDeviceSize>(normalUpload.pixels.size());
    if (totalBytes == 0U
        || totalBytes > static_cast<VkDeviceSize>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Material texture staging size is invalid");
    }
    for (VkBufferImageCopy& region : normalUpload.regions) {
        region.bufferOffset += normalOffset;
    }
    std::vector<std::uint8_t> stagingBytes;
    stagingBytes.reserve(static_cast<std::size_t>(totalBytes));
    stagingBytes.insert(stagingBytes.end(), albedoUpload.pixels.begin(),
                        albedoUpload.pixels.end());
    stagingBytes.insert(stagingBytes.end(), normalUpload.pixels.begin(),
                        normalUpload.pixels.end());

    Buffer staging{};
    VkCommandPool uploadPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence uploadFence = VK_NULL_HANDLE;
    bool uploadSubmitted = false;
    const auto releaseUploadResources = [&] {
        if (device != VK_NULL_HANDLE && uploadFence != VK_NULL_HANDLE) {
            vkDestroyFence(device, uploadFence, nullptr);
            uploadFence = VK_NULL_HANDLE;
        }
        if (device != VK_NULL_HANDLE && uploadPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, uploadPool, nullptr);
            uploadPool = VK_NULL_HANDLE;
        }
        destroyBuffer(staging);
    };

    try {
        createBuffer(staging, totalBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        writeBuffer(staging, stagingBytes.data(), stagingBytes.size());

        VkCommandPoolCreateInfo poolCreate =
            vkInitialize<VkCommandPoolCreateInfo>(
                VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
        poolCreate.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCreate.queueFamilyIndex = queueFamily;
        requireVk(vkCreateCommandPool(device, &poolCreate, nullptr, &uploadPool),
                  "vkCreateCommandPool(material upload)");
        VkCommandBufferAllocateInfo allocate =
            vkInitialize<VkCommandBufferAllocateInfo>(
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
        allocate.commandPool = uploadPool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1U;
        requireVk(vkAllocateCommandBuffers(
                      device, &allocate, &commandBuffer),
                  "vkAllocateCommandBuffers(material upload)");
        VkCommandBufferBeginInfo begin =
            vkInitialize<VkCommandBufferBeginInfo>(
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        requireVk(vkBeginCommandBuffer(commandBuffer, &begin),
                  "vkBeginCommandBuffer(material upload)");

        std::array<VkImageMemoryBarrier2, 2> uploadBarriers{};
        const std::array<VkImage, 2> textureImages{{
            materialAlbedo.image, materialNormal.image}};
        for (std::size_t index = 0U; index < textureImages.size(); ++index) {
            VkImageMemoryBarrier2& barrier = uploadBarriers[index];
            barrier = vkInitialize<VkImageMemoryBarrier2>(
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
            barrier.srcAccessMask = VK_ACCESS_2_NONE;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = textureImages[index];
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = kMaterialMipLevels;
            barrier.subresourceRange.layerCount = kMaterialLayerCount;
        }
        VkDependencyInfo uploadDependency = vkInitialize<VkDependencyInfo>(
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        uploadDependency.imageMemoryBarrierCount =
            static_cast<std::uint32_t>(uploadBarriers.size());
        uploadDependency.pImageMemoryBarriers = uploadBarriers.data();
        pipelineBarrier2(commandBuffer, &uploadDependency);

        vkCmdCopyBufferToImage(
            commandBuffer, staging.handle, materialAlbedo.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<std::uint32_t>(albedoUpload.regions.size()),
            albedoUpload.regions.data());
        vkCmdCopyBufferToImage(
            commandBuffer, staging.handle, materialNormal.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<std::uint32_t>(normalUpload.regions.size()),
            normalUpload.regions.data());

        std::array<VkImageMemoryBarrier2, 2> sampleBarriers{};
        for (std::size_t index = 0U; index < textureImages.size(); ++index) {
            VkImageMemoryBarrier2& barrier = sampleBarriers[index];
            barrier = vkInitialize<VkImageMemoryBarrier2>(
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = textureImages[index];
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = kMaterialMipLevels;
            barrier.subresourceRange.layerCount = kMaterialLayerCount;
        }
        VkDependencyInfo sampleDependency = vkInitialize<VkDependencyInfo>(
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        sampleDependency.imageMemoryBarrierCount =
            static_cast<std::uint32_t>(sampleBarriers.size());
        sampleDependency.pImageMemoryBarriers = sampleBarriers.data();
        pipelineBarrier2(commandBuffer, &sampleDependency);
        requireVk(vkEndCommandBuffer(commandBuffer),
                  "vkEndCommandBuffer(material upload)");

        VkFenceCreateInfo fenceCreate = vkInitialize<VkFenceCreateInfo>(
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
        requireVk(vkCreateFence(device, &fenceCreate, nullptr, &uploadFence),
                  "vkCreateFence(material upload)");
        VkCommandBufferSubmitInfo commandSubmit =
            vkInitialize<VkCommandBufferSubmitInfo>(
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO);
        commandSubmit.commandBuffer = commandBuffer;
        VkSubmitInfo2 submit = vkInitialize<VkSubmitInfo2>(
            VK_STRUCTURE_TYPE_SUBMIT_INFO_2);
        submit.commandBufferInfoCount = 1U;
        submit.pCommandBufferInfos = &commandSubmit;
        requireVk(queueSubmit2(queue, 1U, &submit, uploadFence),
                  "vkQueueSubmit2(material upload)");
        uploadSubmitted = true;
        requireVk(vkWaitForFences(device, 1U, &uploadFence, VK_TRUE,
                                  std::numeric_limits<std::uint64_t>::max()),
                  "vkWaitForFences(material upload)");
        uploadSubmitted = false;
    } catch (...) {
        if (uploadSubmitted && device != VK_NULL_HANDLE
            && queue != VK_NULL_HANDLE) {
            (void)vkQueueWaitIdle(queue);
        }
        releaseUploadResources();
        throw;
    }
    releaseUploadResources();
}

void Renderer::Impl::createMaterialSampler() {
    VkSamplerCreateInfo samplerCreate = vkInitialize<VkSamplerCreateInfo>(
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);
    samplerCreate.magFilter = VK_FILTER_LINEAR;
    samplerCreate.minFilter = VK_FILTER_LINEAR;
    samplerCreate.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreate.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreate.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreate.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreate.mipLodBias = 0.0F;
    samplerCreate.anisotropyEnable = VK_FALSE;
    samplerCreate.maxAnisotropy = 1.0F;
    samplerCreate.compareEnable = VK_FALSE;
    samplerCreate.minLod = 0.0F;
    samplerCreate.maxLod = static_cast<float>(kMaterialMipLevels - 1U);
    samplerCreate.unnormalizedCoordinates = VK_FALSE;
    requireVk(vkCreateSampler(device, &samplerCreate, nullptr,
                              &materialSampler),
              "vkCreateSampler(material textures)");
}

void Renderer::Impl::createFrameContexts() {
    for (FrameContext& frame : frames) {
        VkCommandPoolCreateInfo poolCreate = vkInitialize<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
        poolCreate.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolCreate.queueFamilyIndex = queueFamily;
        requireVk(vkCreateCommandPool(device, &poolCreate, nullptr,
                                      &frame.commandPool),
                  "vkCreateCommandPool");
        VkCommandBufferAllocateInfo allocate = vkInitialize<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
        allocate.commandPool = frame.commandPool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1U;
        requireVk(vkAllocateCommandBuffers(device, &allocate,
                                           &frame.commandBuffer),
                  "vkAllocateCommandBuffers");
        VkFenceCreateInfo fenceCreate = vkInitialize<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
        fenceCreate.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        requireVk(vkCreateFence(device, &fenceCreate, nullptr, &frame.fence),
                  "vkCreateFence");
        const VkSemaphoreCreateInfo semaphoreCreate = vkInitialize<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
        requireVk(vkCreateSemaphore(device, &semaphoreCreate, nullptr,
                                    &frame.imageAvailable),
                  "vkCreateSemaphore(image available)");
        requireVk(vkCreateSemaphore(device, &semaphoreCreate, nullptr,
                                    &frame.renderFinished),
                  "vkCreateSemaphore(render finished)");
        createBuffer(frame.uniform, sizeof(FrameBlock),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        createBuffer(frame.instances,
                     static_cast<VkDeviceSize>(kInstanceCapacity * sizeof(Instance)),
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        createBuffer(frame.impacts,
                     static_cast<VkDeviceSize>(kImpactCommandCapacity
                                               * sizeof(ImpactCommand)),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        createBuffer(frame.hudVertices,
                     static_cast<VkDeviceSize>(kHudVertexCapacity
                                               * sizeof(HudVertex)),
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        createShadowMap(frame.shadowMap);
    }

    if (timestampsSupported) {
        bool queryPoolsReady = true;
        for (FrameContext& frame : frames) {
            VkQueryPoolCreateInfo queryCreate = vkInitialize<VkQueryPoolCreateInfo>(
                VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
            queryCreate.queryType = VK_QUERY_TYPE_TIMESTAMP;
            queryCreate.queryCount = kTimestampQueryCount;
            if (vkCreateQueryPool(device, &queryCreate, nullptr,
                                  &frame.timestampQueries) != VK_SUCCESS) {
                queryPoolsReady = false;
                break;
            }
        }
        if (!queryPoolsReady) {
            for (FrameContext& frame : frames) {
                if (frame.timestampQueries != VK_NULL_HANDLE) {
                    vkDestroyQueryPool(device, frame.timestampQueries, nullptr);
                    frame.timestampQueries = VK_NULL_HANDLE;
                }
            }
            timestampsSupported = false;
        }
    }
}

void Renderer::Impl::createDescriptors() {
    std::array<VkDescriptorSetLayoutBinding, 4> graphicsBindings{};
    graphicsBindings[0].binding = 0U;
    graphicsBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    graphicsBindings[0].descriptorCount = 1U;
    graphicsBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT
                                  | VK_SHADER_STAGE_FRAGMENT_BIT;
    graphicsBindings[1].binding = 1U;
    graphicsBindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    graphicsBindings[1].descriptorCount = 1U;
    graphicsBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    graphicsBindings[2].binding = 2U;
    graphicsBindings[2].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    graphicsBindings[2].descriptorCount = 1U;
    graphicsBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    graphicsBindings[3].binding = 3U;
    graphicsBindings[3].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    graphicsBindings[3].descriptorCount = 1U;
    graphicsBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo graphicsCreate = vkInitialize<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    graphicsCreate.bindingCount =
        static_cast<std::uint32_t>(graphicsBindings.size());
    graphicsCreate.pBindings = graphicsBindings.data();
    requireVk(vkCreateDescriptorSetLayout(device, &graphicsCreate, nullptr,
                                           &graphicsSetLayout),
              "vkCreateDescriptorSetLayout(graphics)");

    std::array<VkDescriptorSetLayoutBinding, 2> storageBindings{};
    for (std::uint32_t binding = 0U;
         binding < static_cast<std::uint32_t>(storageBindings.size());
         ++binding) {
        storageBindings[binding].binding = binding;
        storageBindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBindings[binding].descriptorCount = 1U;
        storageBindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo computeCreate = vkInitialize<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    computeCreate.bindingCount =
        static_cast<std::uint32_t>(storageBindings.size());
    computeCreate.pBindings = storageBindings.data();
    requireVk(vkCreateDescriptorSetLayout(device, &computeCreate, nullptr,
                                           &computeSetLayout),
              "vkCreateDescriptorSetLayout(compute)");

    constexpr std::array<VkDescriptorPoolSize, 3> poolSizes{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFrameCount},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kFrameCount * 3U},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kFrameCount * 2U},
    }};
    VkDescriptorPoolCreateInfo poolCreate = vkInitialize<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    poolCreate.maxSets = kFrameCount * 2U;
    poolCreate.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolCreate.pPoolSizes = poolSizes.data();
    requireVk(vkCreateDescriptorPool(device, &poolCreate, nullptr,
                                      &descriptorPool),
              "vkCreateDescriptorPool");

    std::array<VkDescriptorSetLayout, kFrameCount> graphicsLayouts{};
    graphicsLayouts.fill(graphicsSetLayout);
    std::array<VkDescriptorSet, kFrameCount> graphicsSets{};
    VkDescriptorSetAllocateInfo graphicsAllocate = vkInitialize<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    graphicsAllocate.descriptorPool = descriptorPool;
    graphicsAllocate.descriptorSetCount = kFrameCount;
    graphicsAllocate.pSetLayouts = graphicsLayouts.data();
    requireVk(vkAllocateDescriptorSets(device, &graphicsAllocate,
                                        graphicsSets.data()),
              "vkAllocateDescriptorSets(graphics)");
    for (std::size_t index = 0U; index < frames.size(); ++index) {
        frames[index].graphicsDescriptor = graphicsSets[index];
        const VkDescriptorBufferInfo uniformInfo{
            frames[index].uniform.handle, 0U, sizeof(FrameBlock)};
        const VkDescriptorImageInfo shadowInfo{
            shadowSampler, frames[index].shadowMap.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkDescriptorImageInfo albedoInfo{
            materialSampler, materialAlbedo.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkDescriptorImageInfo normalInfo{
            materialSampler, materialNormal.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        std::array<VkWriteDescriptorSet, 4> writes{};
        writes[0] = vkInitialize<VkWriteDescriptorSet>(
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
        writes[0].dstSet = graphicsSets[index];
        writes[0].dstBinding = 0U;
        writes[0].descriptorCount = 1U;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &uniformInfo;
        writes[1] = vkInitialize<VkWriteDescriptorSet>(
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
        writes[1].dstSet = graphicsSets[index];
        writes[1].dstBinding = 1U;
        writes[1].descriptorCount = 1U;
        writes[1].descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &shadowInfo;
        writes[2] = vkInitialize<VkWriteDescriptorSet>(
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
        writes[2].dstSet = graphicsSets[index];
        writes[2].dstBinding = 2U;
        writes[2].descriptorCount = 1U;
        writes[2].descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo = &albedoInfo;
        writes[3] = vkInitialize<VkWriteDescriptorSet>(
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
        writes[3].dstSet = graphicsSets[index];
        writes[3].dstBinding = 3U;
        writes[3].descriptorCount = 1U;
        writes[3].descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = &normalInfo;
        vkUpdateDescriptorSets(device,
            static_cast<std::uint32_t>(writes.size()), writes.data(), 0U, nullptr);
    }

    std::array<VkDescriptorSetLayout, kFrameCount> computeLayouts{};
    computeLayouts.fill(computeSetLayout);
    std::array<VkDescriptorSet, kFrameCount> computeSets{};
    VkDescriptorSetAllocateInfo computeAllocate = vkInitialize<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    computeAllocate.descriptorPool = descriptorPool;
    computeAllocate.descriptorSetCount = kFrameCount;
    computeAllocate.pSetLayouts = computeLayouts.data();
    requireVk(vkAllocateDescriptorSets(device, &computeAllocate,
                                        computeSets.data()),
              "vkAllocateDescriptorSets(compute)");
    for (std::size_t index = 0U; index < frames.size(); ++index) {
        frames[index].computeDescriptor = computeSets[index];
    }

    VkPipelineLayoutCreateInfo graphicsLayoutCreate = vkInitialize<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    graphicsLayoutCreate.setLayoutCount = 1U;
    graphicsLayoutCreate.pSetLayouts = &graphicsSetLayout;
    requireVk(vkCreatePipelineLayout(device, &graphicsLayoutCreate, nullptr,
                                      &graphicsPipelineLayout),
              "vkCreatePipelineLayout(graphics)");
    const VkPushConstantRange pushRange{
        VK_SHADER_STAGE_COMPUTE_BIT, 0U,
        static_cast<std::uint32_t>(sizeof(SimulationPush))};
    VkPipelineLayoutCreateInfo computeLayoutCreate = vkInitialize<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    computeLayoutCreate.setLayoutCount = 1U;
    computeLayoutCreate.pSetLayouts = &computeSetLayout;
    computeLayoutCreate.pushConstantRangeCount = 1U;
    computeLayoutCreate.pPushConstantRanges = &pushRange;
    requireVk(vkCreatePipelineLayout(device, &computeLayoutCreate, nullptr,
                                      &computePipelineLayout),
              "vkCreatePipelineLayout(compute)");
}

void Renderer::Impl::createMeshes() {
    auto createMesh = [this](Mesh& mesh, auto geometry) {
        auto [vertices, indices] = geometry;
        createBuffer(mesh.vertices,
                     static_cast<VkDeviceSize>(vertices.size() * sizeof(Vertex)),
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        createBuffer(mesh.indices,
                     static_cast<VkDeviceSize>(indices.size()
                                               * sizeof(std::uint16_t)),
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        writeBuffer(mesh.vertices, vertices.data(), vertices.size() * sizeof(Vertex));
        writeBuffer(mesh.indices, indices.data(), indices.size() * sizeof(std::uint16_t));
        mesh.indexCount = static_cast<std::uint32_t>(indices.size());
    };
    createMesh(cube, cubeGeometry());
    // Three shared meshes let projected size, quality, and the flashlight pass
    // choose useful work instead of transforming a single fixed tessellation
    // for every marble and distant warehouse ball.
    createMesh(sphereNear, sphereGeometry(12U, 8U));
    createMesh(sphereMedium, sphereGeometry(8U, 6U));
    createMesh(sphereFar, sphereGeometry(5U, 4U));
}

void Renderer::Impl::createParticles(std::uint32_t seed) {
    const VkDeviceSize maximumBytes =
        static_cast<VkDeviceSize>(kMaximumParticles) * sizeof(Particle);
    const VkDeviceSize maximumRange = properties.limits.maxStorageBufferRange;
    particleCapacity = static_cast<std::uint32_t>(std::min<VkDeviceSize>(
        kMaximumParticles, maximumRange / sizeof(Particle)));
    if (particleCapacity < kSafeParticles) {
        throw std::runtime_error(
            "Selected Vulkan device cannot hold the minimum particle SSBO");
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(particleCapacity)
                             * sizeof(Particle);
    createBuffer(particles, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                  | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    std::vector<Particle> initial(particleCapacity);
    std::mt19937 generator(seed ^ 0x51A6D3U);
    std::uniform_real_distribution<float> xz(0.5F, 99.5F);
    std::uniform_real_distribution<float> y(0.05F, 9.8F);
    std::uniform_real_distribution<float> life(0.1F, 32.0F);
    std::uniform_real_distribution<float> horizontal(-0.65F, 0.65F);
    std::uniform_real_distribution<float> vertical(-0.25F, 1.35F);
    for (std::uint32_t index = 0U; index < particleCapacity; ++index) {
        Particle& particle = initial[index];
        particle.positionLife[0] = xz(generator);
        particle.positionLife[1] = y(generator);
        particle.positionLife[2] = xz(generator);
        particle.positionLife[3] = life(generator);
        particle.velocitySeed[0] = horizontal(generator);
        particle.velocitySeed[1] = vertical(generator);
        particle.velocitySeed[2] = horizontal(generator);
        const std::uint32_t bits =
            (index * 747796405U + 2891336453U) & 0x00ffffffU;
        std::memcpy(&particle.velocitySeed[3], &bits, sizeof(bits));
    }
    writeBuffer(particles, initial.data(),
                static_cast<std::size_t>(bytes));

    for (FrameContext& frame : frames) {
        const VkDescriptorBufferInfo particleInfo{particles.handle, 0U, bytes};
        const VkDescriptorBufferInfo impactInfo{
            frame.impacts.handle, 0U, frame.impacts.size};
        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0] = vkInitialize<VkWriteDescriptorSet>(
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
        writes[0].dstSet = frame.computeDescriptor;
        writes[0].dstBinding = 0U;
        writes[0].descriptorCount = 1U;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &particleInfo;
        writes[1] = vkInitialize<VkWriteDescriptorSet>(
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
        writes[1].dstSet = frame.computeDescriptor;
        writes[1].dstBinding = 1U;
        writes[1].descriptorCount = 1U;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &impactInfo;
        vkUpdateDescriptorSets(device,
            static_cast<std::uint32_t>(writes.size()), writes.data(), 0U, nullptr);
    }

    activeParticles = maximumGpu ? particleCapacity
                                 : std::min(kSafeParticles, particleCapacity);
    (void)maximumBytes;
}

VkShaderModule Renderer::Impl::createShaderModule(std::string_view name) const {
    const std::vector<std::uint32_t> code = readShader(name);
    VkShaderModuleCreateInfo createInfo = vkInitialize<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    createInfo.codeSize = code.size() * sizeof(std::uint32_t);
    createInfo.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    requireVk(vkCreateShaderModule(device, &createInfo, nullptr, &module),
              "vkCreateShaderModule");
    return module;
}

void Renderer::Impl::createComputePipeline() {
    const VkShaderModule shader = createShaderModule("particles.comp.spv");
    VkPipelineShaderStageCreateInfo stage = vkInitialize<VkPipelineShaderStageCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo createInfo = vkInitialize<VkComputePipelineCreateInfo>(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
    createInfo.stage = stage;
    createInfo.layout = computePipelineLayout;
    const VkResult result = vkCreateComputePipelines(
        device, VK_NULL_HANDLE, 1U, &createInfo, nullptr, &computePipeline);
    vkDestroyShaderModule(device, shader, nullptr);
    requireVk(result, "vkCreateComputePipelines");
}

void Renderer::Impl::createShadowPipeline() {
    const VkShaderModule vertex = createShaderModule("shadow.vert.spv");
    VkPipeline pipeline = VK_NULL_HANDLE;
    try {
        VkPipelineShaderStageCreateInfo stage =
            vkInitialize<VkPipelineShaderStageCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
        stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage.module = vertex;
        stage.pName = "main";

        const std::array<VkVertexInputBindingDescription, 2> bindings{{
            {0U, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX},
            {1U, sizeof(Instance), VK_VERTEX_INPUT_RATE_INSTANCE},
        }};
        const std::array<VkVertexInputAttributeDescription, 4> attributes{{
            {0U, 0U, VK_FORMAT_R32G32B32_SFLOAT,
             static_cast<std::uint32_t>(offsetof(Vertex, position))},
            {2U, 1U, VK_FORMAT_R32G32B32A32_SFLOAT,
             static_cast<std::uint32_t>(offsetof(Instance, positionScaleX))},
            {3U, 1U, VK_FORMAT_R32G32B32A32_SFLOAT,
             static_cast<std::uint32_t>(offsetof(Instance, quaternion))},
            {4U, 1U, VK_FORMAT_R32G32B32A32_SFLOAT,
             static_cast<std::uint32_t>(offsetof(Instance, scaleYZMaterial))},
        }};
        VkPipelineVertexInputStateCreateInfo vertexInput =
            vkInitialize<VkPipelineVertexInputStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);
        vertexInput.vertexBindingDescriptionCount =
            static_cast<std::uint32_t>(bindings.size());
        vertexInput.pVertexBindingDescriptions = bindings.data();
        vertexInput.vertexAttributeDescriptionCount =
            static_cast<std::uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();
        VkPipelineInputAssemblyStateCreateInfo assembly =
            vkInitialize<VkPipelineInputAssemblyStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO);
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport =
            vkInitialize<VkPipelineViewportStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO);
        viewport.viewportCount = 1U;
        viewport.scissorCount = 1U;
        VkPipelineRasterizationStateCreateInfo raster =
            vkInitialize<VkPipelineRasterizationStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO);
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_BACK_BIT;
        // perspective() already compensates for Vulkan's downward framebuffer
        // Y axis. Empirical V3D captures and outward-normal winding checks both
        // therefore identify the authored CCW faces as the lit front faces.
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.depthBiasEnable = VK_TRUE;
        raster.lineWidth = 1.0F;
        VkPipelineMultisampleStateCreateInfo multisample =
            vkInitialize<VkPipelineMultisampleStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO);
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth =
            vkInitialize<VkPipelineDepthStencilStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO);
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendStateCreateInfo blending =
            vkInitialize<VkPipelineColorBlendStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO);
        constexpr std::array<VkDynamicState, 3> dynamicStates{{
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_DEPTH_BIAS,
        }};
        VkPipelineDynamicStateCreateInfo dynamic =
            vkInitialize<VkPipelineDynamicStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO);
        dynamic.dynamicStateCount =
            static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkPipelineRenderingCreateInfo rendering =
            vkInitialize<VkPipelineRenderingCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
        rendering.colorAttachmentCount = 0U;
        rendering.depthAttachmentFormat = shadowFormat;
        VkGraphicsPipelineCreateInfo createInfo =
            vkInitialize<VkGraphicsPipelineCreateInfo>(
                VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO);
        createInfo.pNext = &rendering;
        createInfo.stageCount = 1U;
        createInfo.pStages = &stage;
        createInfo.pVertexInputState = &vertexInput;
        createInfo.pInputAssemblyState = &assembly;
        createInfo.pViewportState = &viewport;
        createInfo.pRasterizationState = &raster;
        createInfo.pMultisampleState = &multisample;
        createInfo.pDepthStencilState = &depth;
        createInfo.pColorBlendState = &blending;
        createInfo.pDynamicState = &dynamic;
        createInfo.layout = graphicsPipelineLayout;
        requireVk(vkCreateGraphicsPipelines(
            device, VK_NULL_HANDLE, 1U, &createInfo, nullptr, &pipeline),
            "vkCreateGraphicsPipelines(flashlight shadow)");
    } catch (...) {
        vkDestroyShaderModule(device, vertex, nullptr);
        throw;
    }
    vkDestroyShaderModule(device, vertex, nullptr);
    shadowPipeline = pipeline;
    shadowAvailable = true;
}

void Renderer::Impl::createDepthImage(ImageResource& target) {
    VkImageCreateInfo imageCreate = vkInitialize<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    imageCreate.imageType = VK_IMAGE_TYPE_2D;
    imageCreate.format = depthFormat;
    imageCreate.extent = {extent.width, extent.height, 1U};
    imageCreate.mipLevels = 1U;
    imageCreate.arrayLayers = 1U;
    imageCreate.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreate.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreate.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageCreate.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreate.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    requireVk(vkCreateImage(device, &imageCreate, nullptr, &target.image),
              "vkCreateImage(depth)");
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, target.image, &requirements);
    const auto memoryType = findMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!memoryType) {
        throw std::runtime_error("No device-local memory type for depth image");
    }
    VkMemoryAllocateInfo allocation = vkInitialize<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType->first;
    requireVk(vkAllocateMemory(device, &allocation, nullptr, &target.memory),
              "vkAllocateMemory(depth)");
    requireVk(vkBindImageMemory(device, target.image, target.memory, 0U),
              "vkBindImageMemory(depth)");
    VkImageViewCreateInfo viewCreate = vkInitialize<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
    viewCreate.image = target.image;
    viewCreate.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreate.format = depthFormat;
    viewCreate.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewCreate.subresourceRange.levelCount = 1U;
    viewCreate.subresourceRange.layerCount = 1U;
    requireVk(vkCreateImageView(device, &viewCreate, nullptr, &target.view),
              "vkCreateImageView(depth)");
}

void Renderer::Impl::destroyDepthImage(ImageResource& target) noexcept {
    if (device != VK_NULL_HANDLE && target.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, target.view, nullptr);
    }
    if (device != VK_NULL_HANDLE && target.image != VK_NULL_HANDLE) {
        vkDestroyImage(device, target.image, nullptr);
    }
    if (device != VK_NULL_HANDLE && target.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, target.memory, nullptr);
    }
    target = {};
}

bool Renderer::Impl::createSwapchain() {
    VkSurfaceCapabilitiesKHR capabilities{};
    requireVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                  physicalDevice, surface, &capabilities),
              "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    if (platform == nullptr || platform->width() <= 0 || platform->height() <= 0) {
        return false;
    }

    std::uint32_t formatCount = 0U;
    requireVk(vkGetPhysicalDeviceSurfaceFormatsKHR(
                  physicalDevice, surface, &formatCount, nullptr),
              "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
    if (formatCount == 0U) {
        throw std::runtime_error("The Xlib surface exposes no Vulkan formats");
    }
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    requireVk(vkGetPhysicalDeviceSurfaceFormatsKHR(
                  physicalDevice, surface, &formatCount, formats.data()),
              "vkGetPhysicalDeviceSurfaceFormatsKHR");
    formats.resize(formatCount);
    VkSurfaceFormatKHR chosen = formats.front();
    for (const VkSurfaceFormatKHR& format : formats) {
        if ((format.format == VK_FORMAT_B8G8R8A8_SRGB
             || format.format == VK_FORMAT_R8G8B8A8_SRGB)
            && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = format;
            break;
        }
        if ((chosen.format != VK_FORMAT_B8G8R8A8_SRGB
             && chosen.format != VK_FORMAT_R8G8B8A8_SRGB)
            && (format.format == VK_FORMAT_B8G8R8A8_UNORM
                || format.format == VK_FORMAT_R8G8B8A8_UNORM)) {
            chosen = format;
        }
    }
    colorFormat = chosen.format;
    colorSpace = chosen.colorSpace;

    std::uint32_t presentModeCount = 0U;
    requireVk(vkGetPhysicalDeviceSurfacePresentModesKHR(
                  physicalDevice, surface, &presentModeCount, nullptr),
              "vkGetPhysicalDeviceSurfacePresentModesKHR(count)");
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    if (presentModeCount > 0U) {
        requireVk(vkGetPhysicalDeviceSurfacePresentModesKHR(
                      physicalDevice, surface, &presentModeCount,
                      presentModes.data()),
                  "vkGetPhysicalDeviceSurfacePresentModesKHR");
        presentModes.resize(presentModeCount);
    }
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (maximumGpu) {
        if (std::find(presentModes.begin(), presentModes.end(),
                      VK_PRESENT_MODE_MAILBOX_KHR) != presentModes.end()) {
            presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        } else if (std::find(presentModes.begin(), presentModes.end(),
                             VK_PRESENT_MODE_IMMEDIATE_KHR) != presentModes.end()) {
            presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
    }

    if (capabilities.currentExtent.width
        != std::numeric_limits<std::uint32_t>::max()) {
        extent = capabilities.currentExtent;
    } else {
        const auto requestedWidth = static_cast<std::uint32_t>(platform->width());
        const auto requestedHeight = static_cast<std::uint32_t>(platform->height());
        extent.width = std::clamp(requestedWidth,
                                  capabilities.minImageExtent.width,
                                  capabilities.maxImageExtent.width);
        extent.height = std::clamp(requestedHeight,
                                   capabilities.minImageExtent.height,
                                   capabilities.maxImageExtent.height);
    }
    if (extent.width == 0U || extent.height == 0U) {
        return false;
    }

    std::uint32_t imageCount = capabilities.minImageCount + 1U;
    if (capabilities.maxImageCount > 0U) {
        imageCount = std::min(imageCount, capabilities.maxImageCount);
    }
    VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    constexpr std::array<VkCompositeAlphaFlagBitsKHR, 4> compositeOptions{{
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    }};
    for (VkCompositeAlphaFlagBitsKHR option : compositeOptions) {
        if ((capabilities.supportedCompositeAlpha & option) != 0U) {
            composite = option;
            break;
        }
    }
    VkSwapchainCreateInfoKHR createInfo = vkInitialize<VkSwapchainCreateInfoKHR>(VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = colorFormat;
    createInfo.imageColorSpace = colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1U;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = composite;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    requireVk(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain),
              "vkCreateSwapchainKHR");

    requireVk(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr),
              "vkGetSwapchainImagesKHR(count)");
    swapchainImages.resize(imageCount);
    requireVk(vkGetSwapchainImagesKHR(
                  device, swapchain, &imageCount, swapchainImages.data()),
              "vkGetSwapchainImagesKHR");
    swapchainImages.resize(imageCount);
    swapchainViews.resize(imageCount, VK_NULL_HANDLE);
    swapchainLayouts.resize(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);
    depthImages.resize(imageCount);
    imagesInFlight.resize(imageCount, VK_NULL_HANDLE);
    for (std::uint32_t index = 0U; index < imageCount; ++index) {
        VkImageViewCreateInfo viewCreate = vkInitialize<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
        viewCreate.image = swapchainImages[index];
        viewCreate.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreate.format = colorFormat;
        viewCreate.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreate.subresourceRange.levelCount = 1U;
        viewCreate.subresourceRange.layerCount = 1U;
        requireVk(vkCreateImageView(device, &viewCreate, nullptr,
                                    &swapchainViews[index]),
                  "vkCreateImageView(swapchain)");
        createDepthImage(depthImages[index]);
    }
    createGraphicsPipelines();
    return true;
}

void Renderer::Impl::createGraphicsPipelines() {
    auto createPipeline = [this](std::string_view vertexName,
                                 std::string_view fragmentName,
                                 bool particlesPipeline,
                                 bool hudPipelineKind,
                                 bool resonancePipelineKind) {
        const VkShaderModule vertex = createShaderModule(vertexName);
        VkShaderModule fragment = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        try {
            fragment = createShaderModule(fragmentName);
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertex;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragment;
            stages[1].pName = "main";

            std::array<VkVertexInputBindingDescription, 2> bindings{};
            std::array<VkVertexInputAttributeDescription, 8> attributes{};
            std::uint32_t bindingCount = 1U;
            std::uint32_t attributeCount = 2U;
            if (hudPipelineKind) {
                bindings[0] = {0U, sizeof(HudVertex),
                               VK_VERTEX_INPUT_RATE_VERTEX};
                attributes[0] = {
                    0U, 0U, VK_FORMAT_R32G32_SFLOAT,
                    static_cast<std::uint32_t>(offsetof(HudVertex, position))};
                attributes[1] = {
                    1U, 0U, VK_FORMAT_R32G32B32A32_SFLOAT,
                    static_cast<std::uint32_t>(offsetof(HudVertex, color))};
            } else if (!particlesPipeline) {
                bindings[0] = {0U, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
                bindings[1] = {1U, sizeof(Instance), VK_VERTEX_INPUT_RATE_INSTANCE};
                bindingCount = 2U;
                attributes[0] = {0U, 0U, VK_FORMAT_R32G32B32_SFLOAT,
                                 static_cast<std::uint32_t>(offsetof(Vertex, position))};
                attributes[1] = {1U, 0U, VK_FORMAT_R32G32B32_SFLOAT,
                                 static_cast<std::uint32_t>(offsetof(Vertex, normal))};
                for (std::uint32_t index = 0U; index < 4U; ++index) {
                    attributes[index + 2U] = {
                        index + 2U, 1U, VK_FORMAT_R32G32B32A32_SFLOAT,
                        index * 16U};
                }
                attributes[6] = {
                    6U, 0U, VK_FORMAT_R32G32B32A32_SFLOAT,
                    static_cast<std::uint32_t>(offsetof(Vertex, tangent))};
                attributes[7] = {
                    7U, 0U, VK_FORMAT_R32G32_SFLOAT,
                    static_cast<std::uint32_t>(
                        offsetof(Vertex, textureCoordinate))};
                attributeCount = 8U;
            } else {
                bindings[0] = {0U, sizeof(Particle), VK_VERTEX_INPUT_RATE_VERTEX};
                attributes[0] = {0U, 0U, VK_FORMAT_R32G32B32A32_SFLOAT,
                                 static_cast<std::uint32_t>(offsetof(Particle, positionLife))};
                attributes[1] = {1U, 0U, VK_FORMAT_R32G32B32A32_SFLOAT,
                                 static_cast<std::uint32_t>(offsetof(Particle, velocitySeed))};
            }
            VkPipelineVertexInputStateCreateInfo vertexInput = vkInitialize<VkPipelineVertexInputStateCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);
            vertexInput.vertexBindingDescriptionCount = bindingCount;
            vertexInput.pVertexBindingDescriptions = bindings.data();
            vertexInput.vertexAttributeDescriptionCount = attributeCount;
            vertexInput.pVertexAttributeDescriptions = attributes.data();
            VkPipelineInputAssemblyStateCreateInfo assembly = vkInitialize<VkPipelineInputAssemblyStateCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO);
            assembly.topology = particlesPipeline
                ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewport = vkInitialize<VkPipelineViewportStateCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO);
            viewport.viewportCount = 1U;
            viewport.scissorCount = 1U;
            VkPipelineRasterizationStateCreateInfo raster = vkInitialize<VkPipelineRasterizationStateCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO);
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            const bool opaqueScene = !particlesPipeline && !hudPipelineKind
                                  && !resonancePipelineKind;
            raster.cullMode = opaqueScene ? VK_CULL_MODE_BACK_BIT
                                          : VK_CULL_MODE_NONE;
            // perspective() already compensates for Vulkan framebuffer Y;
            // outward-authored CCW faces remain the lit front faces on V3D.
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0F;
            VkPipelineMultisampleStateCreateInfo multisample = vkInitialize<VkPipelineMultisampleStateCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO);
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo depth = vkInitialize<VkPipelineDepthStencilStateCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO);
            depth.depthTestEnable = hudPipelineKind ? VK_FALSE : VK_TRUE;
            depth.depthWriteEnable =
                (particlesPipeline || hudPipelineKind || resonancePipelineKind)
                    ? VK_FALSE : VK_TRUE;
            depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
                                 | VK_COLOR_COMPONENT_G_BIT
                                 | VK_COLOR_COMPONENT_B_BIT
                                 | VK_COLOR_COMPONENT_A_BIT;
            if (particlesPipeline || resonancePipelineKind) {
                blend.blendEnable = VK_TRUE;
                blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.colorBlendOp = VK_BLEND_OP_ADD;
                blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.alphaBlendOp = VK_BLEND_OP_ADD;
            } else if (hudPipelineKind) {
                blend.blendEnable = VK_TRUE;
                blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                blend.colorBlendOp = VK_BLEND_OP_ADD;
                blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                blend.alphaBlendOp = VK_BLEND_OP_ADD;
            }
            VkPipelineColorBlendStateCreateInfo blending = vkInitialize<VkPipelineColorBlendStateCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO);
            blending.attachmentCount = 1U;
            blending.pAttachments = &blend;
            constexpr std::array<VkDynamicState, 2> dynamicStates{{
                VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}};
            VkPipelineDynamicStateCreateInfo dynamic = vkInitialize<VkPipelineDynamicStateCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO);
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
            dynamic.pDynamicStates = dynamicStates.data();
            VkPipelineRenderingCreateInfo rendering = vkInitialize<VkPipelineRenderingCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
            rendering.colorAttachmentCount = 1U;
            rendering.pColorAttachmentFormats = &colorFormat;
            rendering.depthAttachmentFormat = depthFormat;
            VkGraphicsPipelineCreateInfo createInfo = vkInitialize<VkGraphicsPipelineCreateInfo>(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO);
            createInfo.pNext = &rendering;
            createInfo.stageCount = static_cast<std::uint32_t>(stages.size());
            createInfo.pStages = stages.data();
            createInfo.pVertexInputState = &vertexInput;
            createInfo.pInputAssemblyState = &assembly;
            createInfo.pViewportState = &viewport;
            createInfo.pRasterizationState = &raster;
            createInfo.pMultisampleState = &multisample;
            createInfo.pDepthStencilState = &depth;
            createInfo.pColorBlendState = &blending;
            createInfo.pDynamicState = &dynamic;
            createInfo.layout = graphicsPipelineLayout;
            const VkResult result = vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1U, &createInfo, nullptr, &pipeline);
            requireVk(result, "vkCreateGraphicsPipelines");
        } catch (...) {
            if (fragment != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, fragment, nullptr);
            }
            vkDestroyShaderModule(device, vertex, nullptr);
            throw;
        }
        vkDestroyShaderModule(device, fragment, nullptr);
        vkDestroyShaderModule(device, vertex, nullptr);
        return pipeline;
    };
    scenePipeline = createPipeline(
        "scene.vert.spv", "scene.frag.spv", false, false, false);
    resonancePipeline = createPipeline(
        "resonance.vert.spv", "resonance.frag.spv", false, false, true);
    particlePipeline = createPipeline(
        "particles.vert.spv", "particles.frag.spv", true, false, false);
    hudPipeline = createPipeline(
        "hud.vert.spv", "hud.frag.spv", false, true, false);
}

void Renderer::Impl::destroyGraphicsPipelines() noexcept {
    if (device != VK_NULL_HANDLE && hudPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, hudPipeline, nullptr);
    }
    if (device != VK_NULL_HANDLE && particlePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, particlePipeline, nullptr);
    }
    if (device != VK_NULL_HANDLE && resonancePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, resonancePipeline, nullptr);
    }
    if (device != VK_NULL_HANDLE && scenePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, scenePipeline, nullptr);
    }
    hudPipeline = VK_NULL_HANDLE;
    particlePipeline = VK_NULL_HANDLE;
    resonancePipeline = VK_NULL_HANDLE;
    scenePipeline = VK_NULL_HANDLE;
}

void Renderer::Impl::destroySwapchain() noexcept {
    destroyGraphicsPipelines();
    for (ImageResource& depth : depthImages) {
        destroyDepthImage(depth);
    }
    depthImages.clear();
    if (device != VK_NULL_HANDLE) {
        for (VkImageView view : swapchainViews) {
            if (view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, view, nullptr);
            }
        }
        if (swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device, swapchain, nullptr);
        }
    }
    swapchain = VK_NULL_HANDLE;
    swapchainImages.clear();
    swapchainViews.clear();
    swapchainLayouts.clear();
    imagesInFlight.clear();
    extent = {};
}

bool Renderer::Impl::recreateSwapchain() {
    if (device == VK_NULL_HANDLE || platform == nullptr
        || platform->width() <= 0 || platform->height() <= 0) {
        return false;
    }
    const VkResult idle = vkDeviceWaitIdle(device);
    if (idle == VK_ERROR_DEVICE_LOST) {
        lost = true;
        error = vkFailure("vkDeviceWaitIdle", idle);
        return false;
    }
    if (idle != VK_SUCCESS) {
        error = vkFailure("vkDeviceWaitIdle", idle);
        return false;
    }
    try {
        destroySwapchain();
        return createSwapchain();
    } catch (const std::exception& failure) {
        error = failure.what();
        return false;
    }
}

void Renderer::Impl::appendInstance(std::vector<Instance>& destination,
                                    const Vec3& position,
                                    const Quat& orientation,
                                    const Vec3& scale, const Vec3& color,
                                    float material, float emissive,
                                    float alpha) {
    const std::size_t instanceCount = cubeInstances.size()
        + sphereNearInstances.size() + sphereMediumInstances.size()
        + sphereFarInstances.size() + resonanceInstances.size()
        + shadowCubeInstances.size()
        + shadowSphereNearInstances.size() + shadowSphereMediumInstances.size()
        + shadowSphereFarInstances.size();
    if (instanceCount >= kInstanceCapacity) {
        throw std::runtime_error("PhysicsWorld exceeds Vulkan instance capacity");
    }
    Instance instanceData{};
    instanceData.positionScaleX[0] = gpu(position.x);
    instanceData.positionScaleX[1] = gpu(position.y);
    instanceData.positionScaleX[2] = gpu(position.z);
    instanceData.positionScaleX[3] = gpu(scale.x);
    instanceData.quaternion[0] = gpu(orientation.x);
    instanceData.quaternion[1] = gpu(orientation.y);
    instanceData.quaternion[2] = gpu(orientation.z);
    instanceData.quaternion[3] = gpu(orientation.w);
    instanceData.scaleYZMaterial[0] = gpu(scale.y);
    instanceData.scaleYZMaterial[1] = gpu(scale.z);
    instanceData.scaleYZMaterial[2] = material;
    instanceData.scaleYZMaterial[3] = emissive;
    instanceData.color[0] = gpu(color.x);
    instanceData.color[1] = gpu(color.y);
    instanceData.color[2] = gpu(color.z);
    instanceData.color[3] = alpha;
    destination.push_back(instanceData);
}

void Renderer::Impl::updateScene(const PhysicsWorld& world, float alpha,
                                 FrameContext& frame) {
    cubeInstances.clear();
    sphereNearInstances.clear();
    sphereMediumInstances.clear();
    sphereFarInstances.clear();
    resonanceInstances.clear();
    shadowCubeInstances.clear();
    shadowSphereNearInstances.clear();
    shadowSphereMediumInstances.clear();
    shadowSphereFarInstances.clear();
    hudVerticesCpu.clear();

    const Quat identity{};
    const auto appendRoom = [&](std::vector<Instance>& destination) {
        // Layers 13-15 are reserved for the chamber shell.  Keeping them
        // separate from prop layers 0-12 lets a texture pack replace the room
        // without perturbing the proven material mapping used by rigid bodies.
        appendInstance(destination,
            {kRoomWidth * 0.5, -0.05, kRoomLength * 0.5}, identity,
            {kRoomWidth, 0.10, kRoomLength}, {0.80, 0.86, 0.90}, 13.0F);
        appendInstance(destination,
            {kRoomWidth * 0.5, kRoomHeight + 0.05, kRoomLength * 0.5}, identity,
            {kRoomWidth, 0.10, kRoomLength}, {0.75, 0.80, 0.84}, 15.0F);
        appendInstance(destination,
            {-0.05, kRoomHeight * 0.5, kRoomLength * 0.5}, identity,
            {0.10, kRoomHeight, kRoomLength}, {0.77, 0.84, 0.89}, 14.0F);
        appendInstance(destination,
            {kRoomWidth + 0.05, kRoomHeight * 0.5, kRoomLength * 0.5}, identity,
            {0.10, kRoomHeight, kRoomLength}, {0.77, 0.84, 0.89}, 14.0F);
        appendInstance(destination,
            {kRoomWidth * 0.5, kRoomHeight * 0.5, -0.05}, identity,
            {kRoomWidth, kRoomHeight, 0.10}, {0.77, 0.84, 0.89}, 14.0F);
        appendInstance(destination,
            {kRoomWidth * 0.5, kRoomHeight * 0.5, kRoomLength + 0.05}, identity,
            {kRoomWidth, kRoomHeight, 0.10}, {0.77, 0.84, 0.89}, 14.0F);
    };
    appendRoom(cubeInstances);

    const float amount = clamp(alpha, 0.0F, 1.0F);
    const Player& player = world.player();
    const Real interpolation = static_cast<Real>(amount);
    const Vec3 playerPosition = player.previousPosition * (1.0 - interpolation)
                              + player.position * interpolation;
    const Vec3 eye = playerPosition + Vec3{0.0, kPlayerEyeHeight, 0.0};
    const Vec3 viewForward = player.forward().normalized({0.0, 0.0, 1.0});
    const Vec3 viewSide = viewForward.cross({0.0, 1.0, 0.0})
        .normalized({1.0, 0.0, 0.0});
    const Vec3 viewUp = viewSide.cross(viewForward);
    const Vec3 flashlightOrigin = eye + viewSide * 0.09
                                - viewUp * 0.035
                                + viewForward * 0.06;
    const bool shadowCastersActive = flashlightEnabled
                                  && flashlightShadowsEnabled
                                  && shadowAvailable;
    if (shadowCastersActive) {
        // Shadow casters have their own compact instance ranges. The enclosing
        // room remains present so the flashlight cannot leak through a wall.
        // Nothing is built or uploaded while the pass is inactive.
        appendRoom(shadowCubeInstances);
    }
    const std::size_t qualityIndex = static_cast<std::size_t>(
        clamp(quality, 0, 2));
    constexpr std::array<Real, 3> shadowRanges{{18.0, 25.0, 32.0}};
    const Real shadowRange = shadowRanges[qualityIndex];

    constexpr Real tangentHalfVertical = 0.7265425280053609; // tan(72 deg / 2)
    const Real tangentHalfHorizontal = tangentHalfVertical
        * static_cast<Real>(extent.width) / static_cast<Real>(extent.height);
    const auto cameraVisible = [&](const Vec3& position, Real radius) {
        const Vec3 offset = position - eye;
        const Real depth = offset.dot(viewForward);
        if (depth + radius < 0.025 || depth - radius > 175.0) {
            return false;
        }
        // A small radius margin avoids edge popping while preventing the
        // VideoCore from transforming thousands of off-screen warehouse items.
        const Real projectedDepth = std::max(depth, 0.025);
        return std::abs(offset.dot(viewSide))
                   <= projectedDepth * tangentHalfHorizontal + radius * 1.15
            && std::abs(offset.dot(viewUp))
                   <= projectedDepth * tangentHalfVertical + radius * 1.15;
    };
    constexpr Real tangentHalfFlashlight = 0.6248693519093275; // tan(32 deg)
    const auto flashlightVisible = [&](const Vec3& position, Real radius) {
        const Vec3 offset = position - flashlightOrigin;
        const Real depth = offset.dot(viewForward);
        if (depth + radius < 0.08 || depth - radius > shadowRange) {
            return false;
        }
        const Real projectedDepth = std::max(depth, 0.08);
        const Real coneExtent = projectedDepth * tangentHalfFlashlight
                              + radius * 1.18;
        return std::abs(offset.dot(viewSide)) <= coneExtent
            && std::abs(offset.dot(viewUp)) <= coneExtent;
    };
    constexpr std::array<Real, 3> nearThreshold{{0.070, 0.045, 0.028}};
    constexpr std::array<Real, 3> mediumThreshold{{0.022, 0.012, 0.006}};
    const auto appendSphereLod = [&](bool shadow, const Vec3& position,
                                     const Quat& orientation, Real radius,
                                     const Vec3& color, float material,
                                     float emissive, Real depth,
                                     float instanceAlpha = 1.0F) {
        const Real projectedRadius = radius / std::max(depth, 0.10);
        std::vector<Instance>* destination = nullptr;
        if (projectedRadius >= nearThreshold[qualityIndex]) {
            destination = shadow ? &shadowSphereNearInstances
                                 : &sphereNearInstances;
        } else if (projectedRadius >= mediumThreshold[qualityIndex]) {
            destination = shadow ? &shadowSphereMediumInstances
                                 : &sphereMediumInstances;
        } else {
            destination = shadow ? &shadowSphereFarInstances
                                 : &sphereFarInstances;
        }
        appendInstance(*destination, position, orientation,
                       {radius, radius, radius}, color, material, emissive,
                       instanceAlpha);
    };

    constexpr Real kProbeReach = 14.0;
    probeBody = -1;
    Real probeDistance = kProbeReach + 1.0;
    const auto& specs = world.specs();
    const auto& bodies = world.bodies();
    for (std::size_t bodyIndex = 0U; bodyIndex < bodies.size(); ++bodyIndex) {
        const RigidBody& body = bodies[bodyIndex];
        if (body.spec >= specs.size()) {
            continue;
        }
        const BodySpec& spec = specs[body.spec];
        const Real a = static_cast<Real>(amount);
        const Vec3 position = body.previousPosition * (1.0 - a)
                            + body.position * a;
        const Real radius = std::max(body.cachedBoundingRadius, 0.001);
        if (!body.held) {
            const Real probeRadius = radius
                * (spec.shape == Shape::Sphere ? 1.0 : 1.15);
            const Vec3 rayOffset = eye - position;
            const Real rayProjection = rayOffset.dot(viewForward);
            const Real discriminant = rayProjection * rayProjection
                - (rayOffset.lengthSquared() - probeRadius * probeRadius);
            if (discriminant >= 0.0) {
                const Real root = std::sqrt(discriminant);
                Real distance = -rayProjection - root;
                if (distance < 0.0) {
                    distance = -rayProjection + root;
                }
                if (distance >= 0.0 && distance <= kProbeReach
                    && distance < probeDistance) {
                    probeDistance = distance;
                    probeBody = static_cast<int>(bodyIndex);
                }
            }
        }
        const bool inCamera = cameraVisible(position, radius);
        const bool inFlashlight = shadowCastersActive
                              && flashlightVisible(position, radius);
        if (!inCamera && !inFlashlight) {
            continue;
        }
        const Quat orientation = interpolateQuaternion(
            body.previousOrientation, body.orientation, amount);
        const Vec3 color = body.colorOverride.x >= 0.0
            ? body.colorOverride : spec.color;
        Vec3 renderColor = color;
        float renderEmissive = 0.0F;
        float renderAlpha = 1.0F;
        if (inCamera && hudState.kineticLens) {
            const double mass = static_cast<double>(world.dynamicMass(body));
            const double speedSquared = static_cast<double>(
                body.velocity.lengthSquared());
            const double energy = 0.5 * mass * speedSquared;
            const double heat = clamp(
                std::log10(1.0 + std::max(0.0, energy)) / 4.0, 0.0, 1.0);
            renderColor = kineticHeatColor(energy);
            renderEmissive = static_cast<float>(0.16 + heat * 1.45);
            // A negative instance alpha is an internal scene-shader mode bit;
            // the opaque pipeline still writes alpha=1 to the framebuffer.
            renderAlpha = -1.0F;
        }
        const float material = static_cast<float>(
            static_cast<std::uint8_t>(spec.surfaceMaterial));
        if (spec.shape == Shape::Sphere) {
            const Real renderRadius = std::max(spec.radius(), 0.001);
            if (inCamera) {
                appendSphereLod(false, position, orientation, renderRadius,
                    renderColor, material, renderEmissive,
                    (position - eye).dot(viewForward), renderAlpha);
            }
            if (inFlashlight) {
                appendSphereLod(true, position, orientation, renderRadius,
                    color, material, 0.0F,
                    (position - flashlightOrigin).dot(viewForward));
            }
        } else {
            const Vec3 dimensions{
                std::max(spec.dimensions.x, 0.001),
                std::max(spec.dimensions.y, 0.001),
                std::max(spec.dimensions.z, 0.001),
            };
            if (inCamera) {
                appendInstance(cubeInstances, position, orientation,
                               dimensions, renderColor, material,
                               renderEmissive, renderAlpha);
            }
            if (inFlashlight) {
                appendInstance(shadowCubeInstances, position, orientation,
                               dimensions, color, material);
            }
        }
    }

    if (hudState.echoVisible) {
        const auto appendTrail = [&](std::span<const GalileoTrailSample> trail,
                                     bool previous) {
            if (trail.empty()) {
                return;
            }
            constexpr std::size_t maximumSamples = 96U;
            const std::size_t stride = std::max<std::size_t>(
                1U, (trail.size() + maximumSamples - 1U) / maximumSamples);
            const Real markerRadius = previous ? 0.024 : 0.034;
            const Vec3 steelColor = previous
                ? Vec3{0.25, 0.43, 0.59} : Vec3{0.28, 0.78, 1.0};
            const Vec3 concreteColor = previous
                ? Vec3{0.50, 0.32, 0.19} : Vec3{1.0, 0.54, 0.18};
            const float emissive = previous ? 0.75F : 1.65F;
            for (std::size_t index = 0U; index < trail.size(); index += stride) {
                const GalileoTrailSample& sample = trail[index];
                if (cameraVisible(sample.steel, markerRadius)) {
                    appendInstance(sphereFarInstances, sample.steel, identity,
                        {markerRadius, markerRadius, markerRadius}, steelColor,
                        0.0F, emissive);
                }
                if (cameraVisible(sample.concrete, markerRadius)) {
                    appendInstance(sphereFarInstances, sample.concrete, identity,
                        {markerRadius, markerRadius, markerRadius}, concreteColor,
                        0.0F, emissive);
                }
            }
        };
        appendTrail(world.galileoPreviousTrail(), true);
        appendTrail(world.galileoCurrentTrail(), false);

        const auto appendInterceptTrail = [&cameraVisible, &identity, this](
                std::span<const GravityInterceptTrailSample> trail,
                bool previous) {
            if (trail.empty()) {
                return;
            }
            constexpr std::size_t maximumSamples = 96U;
            const std::size_t stride = std::max<std::size_t>(
                1U, (trail.size() + maximumSamples - 1U) / maximumSamples);
            const Real radius = previous ? 0.025 : 0.037;
            const Vec3 projectileColor = previous
                ? Vec3{0.45, 0.20, 0.46} : Vec3{1.00, 0.20, 0.74};
            const Vec3 targetColor = previous
                ? Vec3{0.15, 0.38, 0.31} : Vec3{0.20, 1.00, 0.58};
            const float emissive = previous ? 0.72F : 1.85F;
            for (std::size_t index = 0U; index < trail.size(); index += stride) {
                const GravityInterceptTrailSample& sample = trail[index];
                if (cameraVisible(sample.projectile, radius)) {
                    appendInstance(sphereFarInstances, sample.projectile,
                        identity, {radius, radius, radius}, projectileColor,
                        0.0F, emissive);
                }
                if (cameraVisible(sample.target, radius)) {
                    appendInstance(sphereFarInstances, sample.target, identity,
                        {radius, radius, radius}, targetColor, 0.0F, emissive);
                }
            }
        };
        appendInterceptTrail(world.gravityInterceptPreviousTrail(), true);
        appendInterceptTrail(world.gravityInterceptCurrentTrail(), false);

        if (const std::optional<Vec3> hit = world.gravityInterceptHitPoint();
            hit.has_value() && cameraVisible(*hit, 0.7)) {
            const Vec3 markerColor{1.00, 0.12, 0.62};
            appendInstance(cubeInstances, *hit, identity,
                           {0.72, 0.035, 0.035}, markerColor, 0.0F, 2.7F);
            appendInstance(cubeInstances, *hit, identity,
                           {0.035, 0.72, 0.035}, markerColor, 0.0F, 2.7F);
            appendInstance(cubeInstances, *hit, identity,
                           {0.035, 0.035, 0.72}, markerColor, 0.0F, 2.7F);
        }
    }

    // Retain only four player-authored Echo Pulses.  Physics exposes the last
    // authoritative event; the renderer snapshots its strongest affected-body
    // impulses once per serial so later frames never depend on transient event
    // storage.  All aging uses simulation time, making F3/F4 freeze/advance the
    // fronts and vector glyphs by the exact fixed tick.
    const double pulseTime = world.simulationTime();
    const std::optional<EchoPulseEvent> latestEchoPulse =
        world.lastEchoPulseEvent();
    if ((lastEchoPulseSimulationTime >= 0.0
         && pulseTime + 1.0e-9 < lastEchoPulseSimulationTime)
        || (!latestEchoPulse.has_value() && lastEchoPulseSerial != 0U)) {
        echoPulseVisuals.fill(EchoPulseVisual{});
        echoPulseVisualCursor = 0U;
        lastEchoPulseSerial = 0U;
    }
    if (latestEchoPulse.has_value()
        && latestEchoPulse->serial != lastEchoPulseSerial) {
        EchoPulseVisual& visual = echoPulseVisuals[
            echoPulseVisualCursor % echoPulseVisuals.size()];
        visual = EchoPulseVisual{};
        visual.serial = latestEchoPulse->serial;
        visual.origin = latestEchoPulse->origin;
        visual.startTime = latestEchoPulse->simulationTime;
        visual.influenceRadius = latestEchoPulse->radius;
        visual.affectedBodyCount = latestEchoPulse->affectedBodyCount;
        visual.totalDeliveredImpulse = latestEchoPulse->totalDeliveredImpulse;

        std::vector<EchoPulseBodyEvent> strongest(
            world.lastEchoPulseBodies().begin(),
            world.lastEchoPulseBodies().end());
        std::sort(strongest.begin(), strongest.end(),
            [](const EchoPulseBodyEvent& left,
               const EchoPulseBodyEvent& right) {
                const Real leftStrength = left.impulse.lengthSquared();
                const Real rightStrength = right.impulse.lengthSquared();
                return leftStrength != rightStrength
                    ? leftStrength > rightStrength : left.body < right.body;
            });
        visual.glyphCount = std::min(
            strongest.size(), visual.glyphs.size());
        for (std::size_t index = 0U; index < visual.glyphCount; ++index) {
            visual.glyphs[index] = {
                strongest[index].body,
                strongest[index].position,
                strongest[index].impulse,
            };
        }
        ++echoPulseVisualCursor;
        lastEchoPulseSerial = latestEchoPulse->serial;
    }
    lastEchoPulseSimulationTime = pulseTime;

    constexpr double echoPulseLifetime = 1.15;
    std::array<const EchoPulseVisual*, kEchoPulseVisualCapacity>
        activeEchoPulses{};
    std::size_t activeEchoPulseCount = 0U;
    for (const EchoPulseVisual& visual : echoPulseVisuals) {
        const double age = pulseTime - visual.startTime;
        if (visual.serial != 0U && age >= 0.0 && age < echoPulseLifetime) {
            activeEchoPulses[activeEchoPulseCount++] = &visual;
        }
    }
    std::sort(activeEchoPulses.begin(),
              activeEchoPulses.begin()
                  + static_cast<std::ptrdiff_t>(activeEchoPulseCount),
              [](const EchoPulseVisual* left, const EchoPulseVisual* right) {
                  return left->startTime > right->startTime;
              });
    constexpr std::array<std::size_t, 3> echoPulseEventCaps{{2U, 3U, 4U}};
    activeEchoPulseCount = std::min(
        activeEchoPulseCount, echoPulseEventCaps[qualityIndex]);
    constexpr std::array<std::size_t, 3> echoPulseGlyphCaps{{16U, 32U, 64U}};
    std::size_t glyphBudget = echoPulseGlyphCaps[qualityIndex];
    std::size_t visibleEchoPulseGlyphs = 0U;
    for (std::size_t eventIndex = 0U;
         eventIndex < activeEchoPulseCount; ++eventIndex) {
        const EchoPulseVisual& visual = *activeEchoPulses[eventIndex];
        const double age = pulseTime - visual.startTime;
        const double fade = std::pow(
            clamp(1.0 - age / echoPulseLifetime, 0.0, 1.0), 1.18);
        const Real influenceRadius = std::max(visual.influenceRadius, 0.25);
        const Real eyeDistance = (eye - visual.origin).length();
        const float strength = static_cast<float>(clamp(
            0.58 + std::log1p(static_cast<double>(
                std::max(visual.totalDeliveredImpulse, 0.0))) * 0.18,
            0.58, 1.65));

        // Cyan is the authoritative outward sweep.  It stops at the exact
        // physics influence radius and glows there briefly; a slower indigo
        // echo follows just inside it to make the finite field unmistakable.
        const Real primaryRadius = std::min(
            influenceRadius, 0.12 + static_cast<Real>(age * 21.5));
        if (cameraVisible(visual.origin, primaryRadius)
            && !(primaryRadius > 6.0
                 && eyeDistance < primaryRadius * 0.72)) {
            appendInstance(resonanceInstances, visual.origin, identity,
                           {primaryRadius, primaryRadius, primaryRadius},
                           {0.08, 0.84, 1.00},
                           static_cast<float>(age / echoPulseLifetime),
                           strength, static_cast<float>(fade * 0.56));
        }
        constexpr double echoDelay = 0.065;
        const double echoAge = age - echoDelay;
        if (echoAge >= 0.0) {
            const Real echoRadius = std::min(
                influenceRadius * 0.975,
                0.09 + static_cast<Real>(echoAge * 16.0));
            if (cameraVisible(visual.origin, echoRadius)
                && !(echoRadius > 6.0
                     && eyeDistance < echoRadius * 0.72)) {
                appendInstance(resonanceInstances, visual.origin, identity,
                               {echoRadius, echoRadius, echoRadius},
                               {0.25, 0.18, 1.00},
                               static_cast<float>(age / echoPulseLifetime),
                               strength * 0.78F,
                               static_cast<float>(fade * 0.30));
            }
        }

        const std::size_t glyphCount = std::min(
            visual.glyphCount, glyphBudget);
        for (std::size_t glyphIndex = 0U;
             glyphIndex < glyphCount; ++glyphIndex) {
            const EchoPulseGlyphSnapshot& glyph = visual.glyphs[glyphIndex];
            if (static_cast<std::size_t>(glyph.body) >= bodies.size()) {
                continue;
            }
            const RigidBody& body = bodies[static_cast<std::size_t>(glyph.body)];
            const Vec3 velocity = body.velocity;
            const Real speed = velocity.length();
            if (speed < 0.20) {
                continue;
            }
            const Vec3 position = body.previousPosition * (1.0 - interpolation)
                                + body.position * interpolation;
            const Vec3 direction = velocity / speed;
            const Real vectorLength = clamp(
                0.18 + std::log1p(speed) * 0.42, 0.22, 2.35);
            const Vec3 tip = position + direction * vectorLength;
            const Vec3 midpoint = (position + tip) * 0.5;
            if (!cameraVisible(midpoint, vectorLength * 0.65)) {
                continue;
            }
            const Real impulseMagnitude = glyph.impulse.length();
            const Real thickness = clamp(
                0.018 + std::log1p(impulseMagnitude) * 0.007,
                0.018, 0.052);
            const Vec3 shaftColor = Vec3{0.06, 0.76, 1.00} * fade;
            const Vec3 tipColor = Vec3{0.38, 0.24, 1.00} * fade;
            const float emissive = static_cast<float>(0.65 + fade * 2.15);
            appendInstance(cubeInstances, midpoint, orientXAxis(direction),
                           {vectorLength, thickness, thickness}, shaftColor,
                           0.0F, emissive);
            const Real tipRadius = thickness * 1.85;
            appendSphereLod(false, tip, identity, tipRadius, tipColor,
                            0.0F, emissive,
                            (tip - eye).dot(viewForward));
            ++visibleEchoPulseGlyphs;
        }
        glyphBudget -= glyphCount;
    }
    const std::size_t visibleEchoPulseFronts = resonanceInstances.size();

    const double resonanceTime = world.simulationTime();
    if (lastResonanceSimulationTime >= 0.0
        && resonanceTime + 1.0e-9 < lastResonanceSimulationTime) {
        resonancePulses.fill(ResonancePulse{});
        resonancePulseCursor = 0U;
    }
    if (resonanceTime != lastResonanceSimulationTime) {
        if (hudState.resonanceVisible) {
            struct Candidate {
                Vec3 position{};
                double response{};
                std::uint32_t family{};
            };
            std::array<Candidate, 3> strongest{};
            std::size_t strongestCount = 0U;
            for (const ImpactEvent& impact : world.impacts()) {
                const double energy = std::max(
                    0.0, 0.5 * static_cast<double>(impact.mass)
                       * static_cast<double>(impact.speed)
                       * static_cast<double>(impact.speed));
                const double response = std::sqrt(energy)
                    + std::log1p(std::max(
                        0.0, static_cast<double>(impact.impulse)));
                if (response < 1.25) {
                    continue;
                }
                Candidate candidate{
                    impact.position, response, impactFamily(impact.family)};
                if (strongestCount < strongest.size()) {
                    strongest[strongestCount++] = candidate;
                } else if (candidate.response > strongest.back().response) {
                    strongest.back() = candidate;
                } else {
                    continue;
                }
                std::sort(strongest.begin(),
                          strongest.begin()
                              + static_cast<std::ptrdiff_t>(strongestCount),
                          [](const Candidate& left, const Candidate& right) {
                              return left.response > right.response;
                          });
            }
            for (std::size_t index = 0U; index < strongestCount; ++index) {
                ResonancePulse& pulse = resonancePulses[
                    resonancePulseCursor % resonancePulses.size()];
                pulse.position = strongest[index].position;
                pulse.startTime = resonanceTime;
                pulse.strength = static_cast<float>(clamp(
                    strongest[index].response * 0.09, 0.32, 1.65));
                pulse.family = strongest[index].family;
                ++resonancePulseCursor;
            }
        }
        lastResonanceSimulationTime = resonanceTime;
    }

    if (hudState.resonanceVisible) {
        std::array<const ResonancePulse*, kResonancePulseCapacity> active{};
        std::size_t activeCount = 0U;
        constexpr double lifetime = 3.8;
        for (const ResonancePulse& pulse : resonancePulses) {
            const double age = resonanceTime - pulse.startTime;
            if (pulse.startTime >= 0.0 && age >= 0.0 && age < lifetime) {
                active[activeCount++] = &pulse;
            }
        }
        std::sort(active.begin(),
                  active.begin() + static_cast<std::ptrdiff_t>(activeCount),
                  [](const ResonancePulse* left, const ResonancePulse* right) {
                      return left->startTime > right->startTime;
                  });
        constexpr std::array<std::size_t, 3> qualityCaps{{4U, 8U, 12U}};
        activeCount = std::min(activeCount, qualityCaps[qualityIndex]);
        for (std::size_t index = 0U; index < activeCount; ++index) {
            const ResonancePulse& pulse = *active[index];
            const double age = resonanceTime - pulse.startTime;
            const Real radius = 0.12 + static_cast<Real>(age * 13.5);
            const Real eyeDistance = (eye - pulse.position).length();
            if (!cameraVisible(pulse.position, radius)
                || (radius > 6.0 && eyeDistance < radius * 0.72)) {
                continue;
            }
            const double fade = std::pow(
                clamp(1.0 - age / lifetime, 0.0, 1.0), 1.35);
            appendInstance(resonanceInstances, pulse.position, identity,
                           {radius, radius, radius},
                           resonanceColor(pulse.family),
                           static_cast<float>(age / lifetime), pulse.strength,
                           static_cast<float>(fade * 0.34));
        }
    }

    const std::array<const std::vector<Instance>*, 9> batches{{
        &cubeInstances, &sphereNearInstances, &sphereMediumInstances,
        &sphereFarInstances, &resonanceInstances, &shadowCubeInstances,
        &shadowSphereNearInstances, &shadowSphereMediumInstances,
        &shadowSphereFarInstances,
    }};
    std::size_t total = 0U;
    for (const auto* batch : batches) {
        total += batch->size();
    }
    if (total > kInstanceCapacity) {
        throw std::runtime_error("PhysicsWorld exceeds Vulkan instance capacity");
    }
    std::size_t instanceOffset = 0U;
    for (const auto* batch : batches) {
        const std::size_t bytes = batch->size() * sizeof(Instance);
        if (bytes > 0U) {
            auto* destination = static_cast<std::byte*>(frame.instances.mapped)
                              + instanceOffset;
            std::memcpy(destination, batch->data(), bytes);
        }
        instanceOffset += bytes;
    }
    if (!frame.instances.coherent && total > 0U) {
        VkMappedMemoryRange range = vkInitialize<VkMappedMemoryRange>(VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
        range.memory = frame.instances.memory;
        range.offset = 0U;
        range.size = VK_WHOLE_SIZE;
        requireVk(vkFlushMappedMemoryRanges(device, 1U, &range),
                  "vkFlushMappedMemoryRanges(instances)");
    }

    const Vec3 target = eye + player.forward();
    const Mat4 projection = perspective(
        72.0F * std::numbers::pi_v<float> / 180.0F,
        static_cast<float>(extent.width) / static_cast<float>(extent.height),
        0.035F, 175.0F);
    const Mat4 viewProjection = multiply(projection, lookAt(eye, target));
    const Mat4 flashlightProjection = perspective(
        64.0F * std::numbers::pi_v<float> / 180.0F,
        1.0F, 0.08F, gpu(shadowRange));
    const Mat4 flashlightViewProjection = multiply(
        flashlightProjection,
        lookAt(flashlightOrigin, flashlightOrigin + viewForward));
    const bool shadowPassActive = shadowCastersActive;
    FrameBlock block{};
    block.viewProjection = viewProjection.value;
    block.cameraFogStart = {gpu(eye.x), gpu(eye.y), gpu(eye.z), 78.0F};
    const std::uint32_t lightCount = overheadLightsEnabled
        ? (quality == 0 ? 2U : quality == 1 ? 3U : 4U)
        : 0U;
    block.fogEndAmbientCountTime = {
        155.0F, overheadLightsEnabled ? 0.54F : 0.10F,
        static_cast<float>(lightCount),
        static_cast<float>(std::fmod(world.simulationTime(), 4096.0))};
    if (overheadLightsEnabled) {
        std::array<std::pair<Real, Vec3>, 100> nearest{};
        std::size_t cursor = 0U;
        for (int z = 0; z < 10; ++z) {
            for (int x = 0; x < 10; ++x) {
                const Vec3 light{5.0 + static_cast<Real>(x) * 10.0,
                                 kRoomHeight - 0.12,
                                 5.0 + static_cast<Real>(z) * 10.0};
                nearest[cursor++] = {(light - eye).lengthSquared(), light};
            }
        }
        std::partial_sort(
            nearest.begin(), nearest.begin() + lightCount, nearest.end(),
            [](const auto& left, const auto& right) {
                return left.first < right.first;
            });
        for (std::size_t index = 0U; index < 4U; ++index) {
            const Vec3 light = nearest[index].second;
            block.lightPosition[index] = {
                gpu(light.x), gpu(light.y), gpu(light.z), 1.0F};
            block.lightColor[index] = {1.0F, 0.985F, 0.91F, 1.0F};
        }
    }
    block.flashlightPositionEnabled = {
        gpu(flashlightOrigin.x), gpu(flashlightOrigin.y),
        gpu(flashlightOrigin.z), flashlightEnabled ? 1.0F : 0.0F};
    block.flashlightDirectionRange = {
        gpu(viewForward.x), gpu(viewForward.y), gpu(viewForward.z),
        gpu(shadowRange)};
    block.flashlightViewProjection = flashlightViewProjection.value;
    block.flashlightShadowParams = {
        static_cast<float>(kShadowQualityResolution[qualityIndex])
            / static_cast<float>(kShadowResolution),
        0.00045F,
        0.0015F,
        shadowPassActive ? (shadowLinearFiltering ? 2.0F : 1.0F) : 0.0F,
    };
    block.materialOptions = {
        materialTexturesEnabled ? 1.0F : 0.0F,
        normalMappingEnabled ? 1.0F : 0.0F,
        hudState.adaptiveNormalDetail
            ? kNormalDetailFullRange[qualityIndex] : 10'000.0F,
        hudState.adaptiveNormalDetail
            ? kNormalDetailCutoff[qualityIndex] : 10'001.0F,
    };
    writeBuffer(frame.uniform, &block, sizeof(block));

    frame.impactCount = 0U;
    frame.impactEmissionCount = 0U;
    frame.impactFirstParticle = 0U;
    frame.impactEndParticle = 0U;
    const double impactTime = world.simulationTime();
    if (impactTime != lastImpactSimulationTime) {
        lastImpactSimulationTime = impactTime;
        std::array<ImpactCommand, kImpactCommandCapacity> commands{};
        constexpr std::array<std::uint32_t, 3> baseEmission{{10U, 18U, 28U}};
        constexpr std::array<std::uint32_t, 3> maximumEmission{{48U, 84U, 132U}};
        for (const ImpactEvent& impact : world.impacts()) {
            if (frame.impactCount >= kImpactCommandCapacity
                || frame.impactEmissionCount >= kImpactEmissionCapacity) {
                break;
            }
            const double visualEnergy = std::max(
                0.0, 0.5 * static_cast<double>(impact.mass)
                   * static_cast<double>(impact.speed)
                   * static_cast<double>(impact.speed));
            const double response = std::sqrt(visualEnergy)
                                  + std::log1p(std::max(
                                        0.0, static_cast<double>(impact.impulse)));
            std::uint32_t emission = baseEmission[qualityIndex]
                + static_cast<std::uint32_t>(clamp(
                    response * (qualityIndex == 0U ? 2.4 : qualityIndex == 1U
                        ? 4.0 : 6.2), 0.0,
                    static_cast<double>(maximumEmission[qualityIndex]
                                             - baseEmission[qualityIndex])));
            emission = std::min(
                emission, kImpactEmissionCapacity - frame.impactEmissionCount);
            ImpactCommand& command = commands[frame.impactCount];
            command.positionStrength[0] = gpu(impact.position.x);
            command.positionStrength[1] = gpu(impact.position.y);
            command.positionStrength[2] = gpu(impact.position.z);
            command.positionStrength[3] = static_cast<float>(clamp(
                response * 0.12 + 0.35, 0.35, 4.0));
            command.metadata[0] = impactFamily(impact.family);
            command.metadata[1] = frame.impactEmissionCount;
            command.metadata[2] = emission;
            command.metadata[3] = frameSerial;
            frame.impactEmissionCount += emission;
            ++frame.impactCount;
        }
        if (frame.impactCount > 0U) {
            const std::uint32_t impactParticlePool = std::min(
                activeParticles, 8'192U);
            if (impactParticlePool > 0U) {
                if (impactParticleCursor + frame.impactEmissionCount
                    > impactParticlePool) {
                    impactParticleCursor = 0U;
                }
                frame.impactFirstParticle = impactParticleCursor;
                frame.impactEndParticle = impactParticleCursor
                                        + frame.impactEmissionCount;
                for (std::uint32_t index = 0U;
                     index < frame.impactCount; ++index) {
                    commands[index].metadata[1] += frame.impactFirstParticle;
                }
                impactParticleCursor = frame.impactEndParticle;
            } else {
                frame.impactCount = 0U;
                frame.impactEmissionCount = 0U;
            }
        }
        if (frame.impactCount > 0U) {
            writeBuffer(frame.impacts, commands.data(),
                static_cast<std::size_t>(frame.impactCount)
                    * sizeof(ImpactCommand));
        }
    }

    if (hudState.visible || hudState.helpVisible) {
        HudBuilder hud{hudVerticesCpu, static_cast<float>(extent.width),
                       static_cast<float>(extent.height)};
        const float scale = clamp(
            static_cast<float>(extent.height) / 480.0F, 1.0F, 2.0F);
        const float line = scale * 9.0F;
        constexpr HudColor panel{0.012F, 0.020F, 0.029F, 0.84F};
        constexpr HudColor panelEdge{0.08F, 0.63F, 0.83F, 0.92F};
        constexpr HudColor primary{0.76F, 0.91F, 0.98F, 0.96F};
        constexpr HudColor secondary{0.49F, 0.69F, 0.78F, 0.94F};
        constexpr HudColor accent{0.19F, 0.84F, 1.0F, 1.0F};
        constexpr HudColor warning{1.0F, 0.62F, 0.18F, 1.0F};
        constexpr HudColor previousEcho{0.55F, 0.57F, 0.62F, 0.94F};

        if (hudState.visible) {
            const bool hasProbe = probeBody >= 0
                && static_cast<std::size_t>(probeBody) < world.bodies().size();
            const HudColor crosshair = hasProbe ? warning : accent;
            const float centerX = static_cast<float>(extent.width) * 0.5F;
            const float centerY = static_cast<float>(extent.height) * 0.5F;
            const float arm = std::max(7.0F, scale * 6.0F);
            const float gap = std::max(3.0F, scale * 2.0F);
            const float thickness = std::max(1.0F, scale);
            hud.quad(centerX - gap - arm, centerY - thickness * 0.5F,
                     arm, thickness, crosshair);
            hud.quad(centerX + gap, centerY - thickness * 0.5F,
                     arm, thickness, crosshair);
            hud.quad(centerX - thickness * 0.5F, centerY - gap - arm,
                     thickness, arm, crosshair);
            hud.quad(centerX - thickness * 0.5F, centerY + gap,
                     thickness, arm, crosshair);
            hud.quad(centerX - thickness, centerY - thickness,
                     thickness * 2.0F, thickness * 2.0F, crosshair);

            const float panelX = 16.0F;
            const float panelY = 16.0F;
            const float panelWidth = std::min(
                static_cast<float>(extent.width) - 32.0F, scale * 330.0F);
            const float panelHeight = line * 24.5F + 24.0F;
            hud.quad(panelX, panelY, panelWidth, panelHeight, panel);
            hud.quad(panelX, panelY, 4.0F, panelHeight, panelEdge);
            float textY = panelY + 10.0F;
            const float textX = panelX + 13.0F;
            hud.text(textX, textY, "NEWTON'S ECHO CHAMBER // LIVE PROBE",
                     scale, accent);
            textY += line * 1.35F;
            constexpr std::array<const char*, 3> qualityNames{{
                "SAFE", "BALANCED", "ULTRA"}};
            const char* simulationState = hudState.singleStep
                ? "SINGLE STEP" : hudState.paused ? "PAUSED" : "RUNNING";
            hud.text(textX, textY, hudFormat(
                "STATE %-11s | %5.1f FPS | %s", simulationState,
                hudState.fps, qualityNames[static_cast<std::size_t>(
                    clamp(hudState.quality, 0, 2))]), scale,
                hudState.paused ? warning : primary);
            textY += line;
            hud.text(textX, textY, hudFormat(
                "GRAVITY %6.2f M/S2 | FRICTION %.2f | THROW %.1f",
                static_cast<double>(world.gravity()),
                static_cast<double>(world.roomFriction()),
                static_cast<double>(world.throwForce())), scale, secondary);
            textY += line;
            hud.text(textX, textY, hudFormat(
                "CONTACTS %zu | CANDIDATES %zu | SOLVER %d",
                world.activeContacts(), world.broadphaseCandidates(),
                world.solverIterations()), scale, secondary);
            textY += line;
            std::string heldName = "NONE";
            const int held = world.heldBody();
            if (held >= 0 && static_cast<std::size_t>(held) < world.bodies().size()) {
                const RigidBody& heldBody = world.bodies()[
                    static_cast<std::size_t>(held)];
                if (heldBody.spec < specs.size()) {
                    heldName = heldBody.instanceLabel.empty()
                        ? specs[heldBody.spec].name : heldBody.instanceLabel;
                }
            }
            hud.text(textX, textY,
                "HELD: " + clippedHudText(heldName, 48U), scale, primary);
            textY += line * 1.25F;

            if (hasProbe) {
                const RigidBody& body = world.bodies()[
                    static_cast<std::size_t>(probeBody)];
                if (body.spec < specs.size()) {
                    const BodySpec& spec = specs[body.spec];
                    const std::string_view name = body.instanceLabel.empty()
                        ? std::string_view(spec.name)
                        : std::string_view(body.instanceLabel);
                    const double mass = static_cast<double>(world.dynamicMass(body));
                    const double speed = static_cast<double>(body.velocity.length());
                    const double energy = 0.5 * mass * speed * speed;
                    hud.text(textX, textY,
                        "TARGET: " + clippedHudText(name, 52U), scale, warning);
                    textY += line;
                    hud.text(textX, textY, hudFormat(
                        "MATERIAL %-17s | MASS %.3f KG",
                        materialName(spec.surfaceMaterial).data(), mass),
                        scale, primary);
                    textY += line;
                    hud.text(textX, textY, hudFormat(
                        "VELOCITY %6.2f M/S | KINETIC %8.2f J", speed, energy),
                        scale, primary);
                    textY += line;
                }
            } else {
                hud.text(textX, textY,
                    "TARGET: NONE WITHIN 14 M", scale, previousEcho);
                textY += line * 2.0F;
            }
            textY += line * 0.25F;
            hud.text(textX, textY,
                "GALILEO: " + clippedHudText(world.galileoStatus(), 50U),
                scale, hudState.echoVisible ? accent : previousEcho);
            textY += line;
            hud.text(textX, textY,
                "TRICK SHOT: "
                    + clippedHudText(world.gravityInterceptStatus(), 46U),
                scale, world.gravityInterceptRunning() ? warning : primary);
            textY += line;
            if (const std::optional<double> hitTime =
                    world.gravityInterceptHitTime(); hitTime.has_value()) {
                hud.text(textX, textY, hudFormat(
                    "INTERCEPT HIT %.3F S | GAP 0.000 M", *hitTime),
                    scale, warning);
            } else if (const std::optional<Real> closestGap =
                           world.gravityInterceptClosestGap();
                       closestGap.has_value()) {
                hud.text(textX, textY, hudFormat(
                    "INTERCEPT T %.2F S | CLOSEST GAP %.3F M",
                    world.gravityInterceptElapsed(),
                    static_cast<double>(*closestGap)), scale, secondary);
            } else {
                hud.text(textX, textY, "INTERCEPT TELEMETRY: STANDBY",
                         scale, previousEcho);
            }
            textY += line;
            hud.text(textX, textY,
                hudState.kineticLens
                    ? "KINETIC LENS ON | LOG J: BLUE > CYAN > GOLD > RED"
                    : "KINETIC LENS OFF | F9 ENABLES LOG-ENERGY VIEW",
                scale, hudState.kineticLens ? accent : previousEcho);
            textY += line;
            const char* textureState = materialTexturesEnabled ? "ON" : "OFF";
            const char* normalState = materialTexturesEnabled
                                   && normalMappingEnabled ? "ON" : "OFF";
            if (hudState.adaptiveNormalDetail) {
                hud.text(textX, textY, hudFormat(
                    "MAT TEX %s | NORMAL %s | ADAPT %.0F-%.0F M",
                    textureState, normalState,
                    static_cast<double>(kNormalDetailFullRange[qualityIndex]),
                    static_cast<double>(kNormalDetailCutoff[qualityIndex])),
                    scale, materialTexturesEnabled && normalMappingEnabled
                        ? secondary : previousEcho);
            } else {
                hud.text(textX, textY, hudFormat(
                    "MAT TEX %s | NORMAL %s | DETAIL FULL",
                    textureState, normalState), scale,
                    materialTexturesEnabled && normalMappingEnabled
                        ? warning : previousEcho);
            }
            textY += line;
            hud.text(textX, textY,
                world.echoPulseReady()
                    ? hudFormat("ECHO PULSE READY | Q EMIT | %zu FRONTS",
                                visibleEchoPulseFronts)
                    : hudFormat("ECHO PULSE COOLDOWN %.2F S | %zu FRONTS",
                                world.echoPulseCooldownRemaining(),
                                visibleEchoPulseFronts),
                scale, world.echoPulseReady() ? accent : warning);
            textY += line;
            if (const std::optional<EchoPulseEvent> pulse =
                    world.lastEchoPulseEvent(); pulse.has_value()) {
                hud.text(textX, textY, hudFormat(
                    "LAST PULSE %u AFFECTED | %.2F N*S | %zu VECTORS",
                    static_cast<unsigned>(pulse->affectedBodyCount),
                    static_cast<double>(pulse->totalDeliveredImpulse),
                    visibleEchoPulseGlyphs), scale, primary);
            } else {
                hud.text(textX, textY,
                    "LAST PULSE: STANDBY | AIM AT A PROP OR OPEN SPACE",
                    scale, previousEcho);
            }
            textY += line;
            hud.text(textX, textY, hudFormat(
                "DRAW %zu | LOD N/M/F %zu/%zu/%zu | SHADOW %zu",
                cubeInstances.size() + sphereNearInstances.size()
                    + sphereMediumInstances.size() + sphereFarInstances.size(),
                sphereNearInstances.size(), sphereMediumInstances.size(),
                sphereFarInstances.size(), shadowCubeInstances.size()
                    + shadowSphereNearInstances.size()
                    + shadowSphereMediumInstances.size()
                    + shadowSphereFarInstances.size()), scale, secondary);
            textY += line;
            const std::size_t impactResonanceShells =
                resonanceInstances.size() >= visibleEchoPulseFronts
                    ? resonanceInstances.size() - visibleEchoPulseFronts : 0U;
            hud.text(textX, textY, hudFormat(
                "RESONANCE %s | %zu IMPACT SHELLS",
                hudState.resonanceVisible ? "ON" : "OFF",
                impactResonanceShells), scale,
                hudState.resonanceVisible ? accent : previousEcho);
            textY += line;
            const char* audioState = !hudState.audioAvailable
                ? "UNAVAILABLE" : hudState.audioMuted ? "MUTED" : "ACTIVE";
            hud.text(textX, textY,
                std::string("AUDIO: ") + audioState, scale,
                !hudState.audioAvailable || hudState.audioMuted
                    ? warning : primary);
            textY += line;
            hud.text(textX, textY,
                "MESSAGE: " + clippedHudText(world.lastMessage(), 50U),
                scale, secondary);

            const std::string footer =
                "F1 HELP  F3 PAUSE  Q PULSE  F11 MAT  F12 DETAIL";
            const float footerWidth = static_cast<float>(footer.size())
                                    * scale * 6.0F + 20.0F;
            const float footerX = std::max(
                10.0F, (static_cast<float>(extent.width) - footerWidth) * 0.5F);
            const float footerY = static_cast<float>(extent.height) - line - 12.0F;
            hud.quad(footerX - 8.0F, footerY - 6.0F,
                     std::min(footerWidth,
                              static_cast<float>(extent.width) - 20.0F),
                     line + 10.0F, panel);
            hud.text(footerX, footerY, footer, scale, secondary);

            if (hudState.paused) {
                const std::string banner = hudState.singleStep
                    ? "SINGLE FIXED STEP" : "SIMULATION PAUSED";
                const float bannerWidth = static_cast<float>(banner.size())
                                        * scale * 6.0F + 28.0F;
                const float bannerX = (static_cast<float>(extent.width)
                                     - bannerWidth) * 0.5F;
                hud.quad(bannerX, 18.0F, bannerWidth, line + 15.0F, panel);
                hud.quad(bannerX, 18.0F, bannerWidth, 3.0F, warning);
                hud.text(bannerX + 14.0F, 26.0F, banner, scale, warning);
            }
        }

        if (hudState.helpVisible) {
            const std::array<std::string_view, 21> helpLines{{
                "NEWTON'S ECHO CHAMBER // FIELD MANUAL",
                "F1  CLOSE THIS HELP",
                "F2  SHOW/HIDE HUD AND LIVE PROBE",
                "F3  PAUSE/RESUME AUTHORITATIVE SIMULATION",
                "F4  SINGLE FIXED STEP WHILE PAUSED",
                "F5  RESET THE CHAMBER",
                "F6  START/RESTART GALILEO DROP EXPERIMENT",
                "F7  SHOW/HIDE CURRENT + PREVIOUS ECHO TRAILS",
                "F8  START/RESTART GRAVITY INTERCEPT TRICK SHOT",
                "F9  TOGGLE LOG-ENERGY KINETIC LENS",
                "F10 TOGGLE MATERIAL-COLORED RESONANCE WAVES",
                "Q   EMIT AUTHORITATIVE 8 M ECHO PULSE",
                "F11 CYCLE MATERIALS: FULL / ALBEDO / UNIFORM",
                "F12 TOGGLE ADAPTIVE / FULL NORMAL DETAIL",
                "WASD MOVE  |  MOUSE LOOK  |  SPACE JUMP",
                "R/F GRAVITY  |  T/G THROW FORCE  |  Y/H FRICTION",
                "RMB PICK UP/DROP  |  LMB THROW  |  C FLASHLIGHT",
                "V OVERHEAD LIGHTS  |  M AUDIO  |  TAB RELEASE MOUSE",
                "BLUE/ORANGE = CURRENT DROP  |  DIM = PREVIOUS DROP",
                "VIOLET/LIME = PROJECTILE/TARGET  |  PINK STAR = HIT",
                "PHYSICS REMAINS FIXED AT 120 HZ WHILE RENDERING ADAPTS",
            }};
            const float helpWidth = std::min(
                static_cast<float>(extent.width) - 40.0F, scale * 390.0F);
            const float helpHeight = line * static_cast<float>(helpLines.size())
                                   + 34.0F;
            const float helpX = (static_cast<float>(extent.width) - helpWidth)
                              * 0.5F;
            const float helpY = std::max(
                20.0F, (static_cast<float>(extent.height) - helpHeight) * 0.5F);
            hud.quad(helpX, helpY, helpWidth, helpHeight,
                     {0.006F, 0.012F, 0.020F, 0.96F});
            hud.quad(helpX, helpY, helpWidth, 4.0F, panelEdge);
            float helpTextY = helpY + 14.0F;
            for (std::size_t index = 0U; index < helpLines.size(); ++index) {
                hud.text(helpX + 16.0F, helpTextY, helpLines[index], scale,
                         index == 0U ? accent : index >= 6U && index <= 13U
                             ? warning : primary);
                helpTextY += line;
            }
        }
    }

    frame.hudVertexCount = static_cast<std::uint32_t>(hudVerticesCpu.size());
    if (!hudVerticesCpu.empty()) {
        writeBuffer(frame.hudVertices, hudVerticesCpu.data(),
                    hudVerticesCpu.size() * sizeof(HudVertex));
    }
    ++frameSerial;
}

void Renderer::Impl::collectFrameTimestamps(FrameContext& frame) noexcept {
    if (!frame.timestampPending || frame.timestampQueries == VK_NULL_HANDLE
        || device == VK_NULL_HANDLE) {
        return;
    }

    std::array<std::uint64_t, kTimestampQueryCount> values{};
    const VkResult result = vkGetQueryPoolResults(
        device, frame.timestampQueries, 0U, kTimestampQueryCount,
        sizeof(values), values.data(), sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT);
    frame.timestampPending = false;
    if (result != VK_SUCCESS || timestampValidBits == 0U
        || timestampPeriodNanoseconds <= 0.0) {
        // Instrumentation must never turn an otherwise valid frame into a
        // renderer failure. Stop issuing timestamps if the implementation
        // cannot return completed results after its submission fence.
        timestampsSupported = false;
        return;
    }

    const std::uint64_t mask = timestampValidBits >= 64U
        ? std::numeric_limits<std::uint64_t>::max()
        : (std::uint64_t{1} << timestampValidBits) - 1U;
    const auto elapsedTicks = [mask](std::uint64_t begin,
                                     std::uint64_t end) noexcept {
        return ((end & mask) - (begin & mask)) & mask;
    };
    const double millisecondsPerTick = timestampPeriodNanoseconds / 1'000'000.0;
    const double graphicsMilliseconds = static_cast<double>(elapsedTicks(
        values[kTimestampGraphicsStart], values[kTimestampGraphicsEnd]))
        * millisecondsPerTick;
    const double shadowMilliseconds = frame.timestampShadowActive
        ? static_cast<double>(elapsedTicks(values[kTimestampShadowStart],
                                           values[kTimestampShadowEnd]))
            * millisecondsPerTick
        : 0.0;
    const double totalMilliseconds = static_cast<double>(elapsedTicks(
        values[kTimestampTotalStart], values[kTimestampGraphicsEnd]))
        * millisecondsPerTick;
    const double computeMilliseconds = frame.timestampComputeActive
        ? static_cast<double>(elapsedTicks(values[kTimestampComputeStart],
                                           values[kTimestampComputeEnd]))
            * millisecondsPerTick
        : 0.0;

    gpuComputeMillisecondsSum += computeMilliseconds;
    gpuShadowMillisecondsSum += shadowMilliseconds;
    gpuGraphicsMillisecondsSum += graphicsMilliseconds;
    gpuTotalMillisecondsSum += totalMilliseconds;
    ++timedFrames;
}

void Renderer::Impl::collectAllFrameTimestamps() noexcept {
    for (FrameContext& frame : frames) {
        collectFrameTimestamps(frame);
    }
}

bool Renderer::Impl::acquireImage(FrameContext& frame,
                                  std::uint32_t& imageIndex,
                                  bool& suboptimal) {
    suboptimal = false;
    for (int attempt = 0; attempt < 2; ++attempt) {
        const VkResult result = vkAcquireNextImageKHR(
            device, swapchain, std::numeric_limits<std::uint64_t>::max(),
            frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
        if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
            suboptimal = result == VK_SUBOPTIMAL_KHR;
            return true;
        }
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            if (!recreateSwapchain()) {
                return false;
            }
            continue;
        }
        if (result == VK_ERROR_DEVICE_LOST) {
            lost = true;
        }
        error = vkFailure("vkAcquireNextImageKHR", result);
        return false;
    }
    error = "Swapchain remained out of date after recreation";
    return false;
}

void Renderer::Impl::record(FrameContext& frame, std::uint32_t imageIndex,
                            const PhysicsWorld& world,
                            float frameDeltaSeconds) {
    requireVk(vkResetCommandPool(device, frame.commandPool, 0U),
              "vkResetCommandPool");
    VkCommandBufferBeginInfo begin = vkInitialize<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    requireVk(vkBeginCommandBuffer(frame.commandBuffer, &begin),
              "vkBeginCommandBuffer");

    const bool computeActive = gpuPhysicsEnabled && activeParticles > 0U
                            && frameDeltaSeconds > 0.0F;
    const bool shadowPassActive = flashlightEnabled
                               && flashlightShadowsEnabled
                               && shadowAvailable
                               && shadowPipeline != VK_NULL_HANDLE
                               && frame.shadowMap.image != VK_NULL_HANDLE;
    const bool timeFrame = timestampsSupported
                        && frame.timestampQueries != VK_NULL_HANDLE;
    if (timeFrame) {
        vkCmdResetQueryPool(frame.commandBuffer, frame.timestampQueries,
                            0U, kTimestampQueryCount);
        vkCmdWriteTimestamp(frame.commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            frame.timestampQueries, kTimestampTotalStart);
    }

    if (computeActive) {
        VkBufferMemoryBarrier2 beforeCompute = vkInitialize<VkBufferMemoryBarrier2>(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2);
        beforeCompute.srcStageMask = renderedFrames == 0U
            ? VK_PIPELINE_STAGE_2_HOST_BIT
            : VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
        beforeCompute.srcAccessMask = renderedFrames == 0U
            ? VK_ACCESS_2_HOST_WRITE_BIT : VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        beforeCompute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        beforeCompute.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                    | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        beforeCompute.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        beforeCompute.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        beforeCompute.buffer = particles.handle;
        beforeCompute.size = particles.size;
        VkBufferMemoryBarrier2 impactBeforeCompute =
            vkInitialize<VkBufferMemoryBarrier2>(
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2);
        impactBeforeCompute.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        impactBeforeCompute.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        impactBeforeCompute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        impactBeforeCompute.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        impactBeforeCompute.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        impactBeforeCompute.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        impactBeforeCompute.buffer = frame.impacts.handle;
        impactBeforeCompute.size = frame.impacts.size;
        const std::array<VkBufferMemoryBarrier2, 2> beforeComputeBarriers{{
            beforeCompute, impactBeforeCompute}};
        VkDependencyInfo dependency = vkInitialize<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        dependency.bufferMemoryBarrierCount =
            static_cast<std::uint32_t>(beforeComputeBarriers.size());
        dependency.pBufferMemoryBarriers = beforeComputeBarriers.data();
        pipelineBarrier2(frame.commandBuffer, &dependency);

        if (timeFrame) {
            vkCmdWriteTimestamp(frame.commandBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                frame.timestampQueries, kTimestampComputeStart);
        }

        gpuTime = std::fmod(gpuTime
            + static_cast<double>(clamp(frameDeltaSeconds, 0.0F, 0.05F)), 4096.0);
        const SimulationPush push{
            activeParticles,
            clamp(frameDeltaSeconds, 0.0F, 0.05F),
            gpu(world.gravity()),
            gpu(world.roomFriction()),
            static_cast<float>(gpuTime),
            frame.impactCount,
            frame.impactEmissionCount,
            frame.impactFirstParticle,
            frame.impactEndParticle,
            frameSerial,
        };
        vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          computePipeline);
        vkCmdBindDescriptorSets(frame.commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout,
            0U, 1U, &frame.computeDescriptor, 0U, nullptr);
        vkCmdPushConstants(frame.commandBuffer, computePipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<std::uint32_t>(sizeof(push)), &push);
        vkCmdDispatch(frame.commandBuffer,
                      (activeParticles + kParticleLocalSize - 1U)
                        / kParticleLocalSize,
                      1U, 1U);

        if (timeFrame) {
            vkCmdWriteTimestamp(frame.commandBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                frame.timestampQueries, kTimestampComputeEnd);
        }

        VkBufferMemoryBarrier2 beforeVertex = vkInitialize<VkBufferMemoryBarrier2>(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2);
        beforeVertex.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        beforeVertex.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        beforeVertex.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
        beforeVertex.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        beforeVertex.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        beforeVertex.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        beforeVertex.buffer = particles.handle;
        beforeVertex.size = particles.size;
        dependency.bufferMemoryBarrierCount = 1U;
        dependency.pBufferMemoryBarriers = &beforeVertex;
        pipelineBarrier2(frame.commandBuffer, &dependency);
    } else {
        if (gpuPhysicsEnabled && activeParticles > 0U && renderedFrames == 0U) {
            VkBufferMemoryBarrier2 particlesBeforeVertex =
                vkInitialize<VkBufferMemoryBarrier2>(
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2);
            particlesBeforeVertex.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
            particlesBeforeVertex.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
            particlesBeforeVertex.dstStageMask =
                VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
            particlesBeforeVertex.dstAccessMask =
                VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
            particlesBeforeVertex.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            particlesBeforeVertex.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            particlesBeforeVertex.buffer = particles.handle;
            particlesBeforeVertex.size = particles.size;
            VkDependencyInfo particlesDependency =
                vkInitialize<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
            particlesDependency.bufferMemoryBarrierCount = 1U;
            particlesDependency.pBufferMemoryBarriers = &particlesBeforeVertex;
            pipelineBarrier2(frame.commandBuffer, &particlesDependency);
        }
        if (timeFrame) {
            // Keep every timed frame's query layout complete. Collection reports
            // compute as exactly zero when the secondary simulation is disabled.
            vkCmdWriteTimestamp(frame.commandBuffer,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                frame.timestampQueries, kTimestampComputeStart);
            vkCmdWriteTimestamp(frame.commandBuffer,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                frame.timestampQueries, kTimestampComputeEnd);
        }
    }

    auto drawMesh = [&](const Mesh& mesh, std::uint32_t instanceCount,
                        VkDeviceSize instanceOffset) {
        if (instanceCount == 0U) {
            return;
        }
        const std::array<VkBuffer, 2> buffers{{
            mesh.vertices.handle, frame.instances.handle}};
        const std::array<VkDeviceSize, 2> offsets{{0U, instanceOffset}};
        vkCmdBindVertexBuffers(frame.commandBuffer, 0U,
                               static_cast<std::uint32_t>(buffers.size()),
                               buffers.data(), offsets.data());
        vkCmdBindIndexBuffer(frame.commandBuffer, mesh.indices.handle,
                             0U, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(frame.commandBuffer, mesh.indexCount,
                         instanceCount, 0U, 0, 0U);
    };
    const auto instanceBytes = [](const std::vector<Instance>& values) {
        return static_cast<VkDeviceSize>(values.size() * sizeof(Instance));
    };
    const VkDeviceSize cameraCubeOffset = 0U;
    const VkDeviceSize cameraNearOffset =
        cameraCubeOffset + instanceBytes(cubeInstances);
    const VkDeviceSize cameraMediumOffset =
        cameraNearOffset + instanceBytes(sphereNearInstances);
    const VkDeviceSize cameraFarOffset =
        cameraMediumOffset + instanceBytes(sphereMediumInstances);
    const VkDeviceSize resonanceOffset =
        cameraFarOffset + instanceBytes(sphereFarInstances);
    const VkDeviceSize shadowCubeOffset =
        resonanceOffset + instanceBytes(resonanceInstances);
    const VkDeviceSize shadowNearOffset =
        shadowCubeOffset + instanceBytes(shadowCubeInstances);
    const VkDeviceSize shadowMediumOffset =
        shadowNearOffset + instanceBytes(shadowSphereNearInstances);
    const VkDeviceSize shadowFarOffset =
        shadowMediumOffset + instanceBytes(shadowSphereMediumInstances);
    const std::uint32_t shadowResolution = kShadowQualityResolution[
        static_cast<std::size_t>(clamp(quality, 0, 2))];

    if (shadowPassActive) {
        if (timeFrame) {
            vkCmdWriteTimestamp(frame.commandBuffer,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                frame.timestampQueries, kTimestampShadowStart);
        }
        VkImageMemoryBarrier2 shadowAttachmentBarrier =
            vkInitialize<VkImageMemoryBarrier2>(
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
        shadowAttachmentBarrier.srcStageMask =
            frame.shadowMap.layout == VK_IMAGE_LAYOUT_UNDEFINED
                ? VK_PIPELINE_STAGE_2_NONE
                : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        shadowAttachmentBarrier.srcAccessMask =
            frame.shadowMap.layout == VK_IMAGE_LAYOUT_UNDEFINED
                ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        shadowAttachmentBarrier.dstStageMask =
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        shadowAttachmentBarrier.dstAccessMask =
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        shadowAttachmentBarrier.oldLayout = frame.shadowMap.layout;
        shadowAttachmentBarrier.newLayout =
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        shadowAttachmentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        shadowAttachmentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        shadowAttachmentBarrier.image = frame.shadowMap.image;
        shadowAttachmentBarrier.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_DEPTH_BIT;
        shadowAttachmentBarrier.subresourceRange.levelCount = 1U;
        shadowAttachmentBarrier.subresourceRange.layerCount = 1U;
        VkDependencyInfo shadowAttachmentDependency =
            vkInitialize<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        shadowAttachmentDependency.imageMemoryBarrierCount = 1U;
        shadowAttachmentDependency.pImageMemoryBarriers =
            &shadowAttachmentBarrier;
        pipelineBarrier2(frame.commandBuffer, &shadowAttachmentDependency);

        VkRenderingAttachmentInfo shadowDepthAttachment =
            vkInitialize<VkRenderingAttachmentInfo>(
                VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
        shadowDepthAttachment.imageView = frame.shadowMap.view;
        shadowDepthAttachment.imageLayout =
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        shadowDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        shadowDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        shadowDepthAttachment.clearValue.depthStencil = {1.0F, 0U};
        VkRenderingInfo shadowRendering = vkInitialize<VkRenderingInfo>(
            VK_STRUCTURE_TYPE_RENDERING_INFO);
        shadowRendering.renderArea.extent = {
            shadowResolution, shadowResolution};
        shadowRendering.layerCount = 1U;
        shadowRendering.colorAttachmentCount = 0U;
        shadowRendering.pDepthAttachment = &shadowDepthAttachment;
        beginRendering(frame.commandBuffer, &shadowRendering);
        const VkViewport shadowViewport{
            0.0F, 0.0F,
            static_cast<float>(shadowResolution),
            static_cast<float>(shadowResolution),
            0.0F, 1.0F};
        const VkRect2D shadowScissor{{0, 0},
            {shadowResolution, shadowResolution}};
        vkCmdSetViewport(frame.commandBuffer, 0U, 1U, &shadowViewport);
        vkCmdSetScissor(frame.commandBuffer, 0U, 1U, &shadowScissor);
        vkCmdSetDepthBias(frame.commandBuffer, 1.25F, 0.0F, 1.75F);
        vkCmdBindPipeline(frame.commandBuffer,
                          VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
        vkCmdBindDescriptorSets(frame.commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout,
            0U, 1U, &frame.graphicsDescriptor, 0U, nullptr);
        drawMesh(cube, static_cast<std::uint32_t>(shadowCubeInstances.size()),
                 shadowCubeOffset);
        drawMesh(sphereNear,
                 static_cast<std::uint32_t>(shadowSphereNearInstances.size()),
                 shadowNearOffset);
        drawMesh(sphereMedium,
                 static_cast<std::uint32_t>(shadowSphereMediumInstances.size()),
                 shadowMediumOffset);
        drawMesh(sphereFar,
                 static_cast<std::uint32_t>(shadowSphereFarInstances.size()),
                 shadowFarOffset);
        endRendering(frame.commandBuffer);

        VkImageMemoryBarrier2 shadowSampleBarrier =
            vkInitialize<VkImageMemoryBarrier2>(
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
        shadowSampleBarrier.srcStageMask =
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        shadowSampleBarrier.srcAccessMask =
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        shadowSampleBarrier.dstStageMask =
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        shadowSampleBarrier.dstAccessMask =
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        shadowSampleBarrier.oldLayout =
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        shadowSampleBarrier.newLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        shadowSampleBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        shadowSampleBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        shadowSampleBarrier.image = frame.shadowMap.image;
        shadowSampleBarrier.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_DEPTH_BIT;
        shadowSampleBarrier.subresourceRange.levelCount = 1U;
        shadowSampleBarrier.subresourceRange.layerCount = 1U;
        VkDependencyInfo shadowSampleDependency =
            vkInitialize<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        shadowSampleDependency.imageMemoryBarrierCount = 1U;
        shadowSampleDependency.pImageMemoryBarriers = &shadowSampleBarrier;
        pipelineBarrier2(frame.commandBuffer, &shadowSampleDependency);
        frame.shadowMap.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (timeFrame) {
            vkCmdWriteTimestamp(frame.commandBuffer,
                                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                frame.timestampQueries, kTimestampShadowEnd);
        }
    } else {
        // Binding 1 is statically present in the scene shader even when the
        // runtime shadow branch is disabled. Give a never-rendered map the
        // descriptor-declared read layout before its first scene draw so core
        // validation never observes UNDEFINED at a sampled binding.
        if (frame.shadowMap.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            VkImageMemoryBarrier2 initializeShadowRead =
                vkInitialize<VkImageMemoryBarrier2>(
                    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
            initializeShadowRead.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
            initializeShadowRead.srcAccessMask = VK_ACCESS_2_NONE;
            initializeShadowRead.dstStageMask =
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            initializeShadowRead.dstAccessMask =
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            initializeShadowRead.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            initializeShadowRead.newLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            initializeShadowRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            initializeShadowRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            initializeShadowRead.image = frame.shadowMap.image;
            initializeShadowRead.subresourceRange.aspectMask =
                VK_IMAGE_ASPECT_DEPTH_BIT;
            initializeShadowRead.subresourceRange.levelCount = 1U;
            initializeShadowRead.subresourceRange.layerCount = 1U;
            VkDependencyInfo initializeShadowDependency =
                vkInitialize<VkDependencyInfo>(
                    VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
            initializeShadowDependency.imageMemoryBarrierCount = 1U;
            initializeShadowDependency.pImageMemoryBarriers =
                &initializeShadowRead;
            pipelineBarrier2(frame.commandBuffer, &initializeShadowDependency);
            frame.shadowMap.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        if (timeFrame) {
            vkCmdWriteTimestamp(frame.commandBuffer,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                frame.timestampQueries, kTimestampShadowStart);
            vkCmdWriteTimestamp(frame.commandBuffer,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                frame.timestampQueries, kTimestampShadowEnd);
        }
    }

    VkImageMemoryBarrier2 colorBarrier = vkInitialize<VkImageMemoryBarrier2>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
    colorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    colorBarrier.srcAccessMask = VK_ACCESS_2_NONE;
    colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    colorBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    colorBarrier.oldLayout = swapchainLayouts[imageIndex];
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorBarrier.image = swapchainImages[imageIndex];
    colorBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorBarrier.subresourceRange.levelCount = 1U;
    colorBarrier.subresourceRange.layerCount = 1U;
    VkImageMemoryBarrier2 depthBarrier = vkInitialize<VkImageMemoryBarrier2>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
    depthBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    depthBarrier.srcAccessMask = VK_ACCESS_2_NONE;
    depthBarrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                              | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    depthBarrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                               | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthBarrier.oldLayout = depthImages[imageIndex].initialized
        ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.image = depthImages[imageIndex].image;
    depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthBarrier.subresourceRange.levelCount = 1U;
    depthBarrier.subresourceRange.layerCount = 1U;
    std::array<VkImageMemoryBarrier2, 2> attachmentBarriers{{
        colorBarrier, depthBarrier}};
    VkDependencyInfo attachmentsDependency = vkInitialize<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
    attachmentsDependency.imageMemoryBarrierCount =
        static_cast<std::uint32_t>(attachmentBarriers.size());
    attachmentsDependency.pImageMemoryBarriers = attachmentBarriers.data();
    pipelineBarrier2(frame.commandBuffer, &attachmentsDependency);

    const VkClearValue clearColor{{{0.025F, 0.035F, 0.047F, 1.0F}}};
    VkRenderingAttachmentInfo colorAttachment = vkInitialize<VkRenderingAttachmentInfo>(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
    colorAttachment.imageView = swapchainViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearColor;
    VkRenderingAttachmentInfo depthAttachment = vkInitialize<VkRenderingAttachmentInfo>(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
    depthAttachment.imageView = depthImages[imageIndex].view;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue.depthStencil = {1.0F, 0U};
    VkRenderingInfo rendering = vkInitialize<VkRenderingInfo>(VK_STRUCTURE_TYPE_RENDERING_INFO);
    rendering.renderArea.extent = extent;
    rendering.layerCount = 1U;
    rendering.colorAttachmentCount = 1U;
    rendering.pColorAttachments = &colorAttachment;
    rendering.pDepthAttachment = &depthAttachment;
    if (timeFrame) {
        vkCmdWriteTimestamp(frame.commandBuffer,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            frame.timestampQueries, kTimestampGraphicsStart);
    }
    beginRendering(frame.commandBuffer, &rendering);

    const VkViewport viewport{0.0F, 0.0F,
        static_cast<float>(extent.width), static_cast<float>(extent.height),
        0.0F, 1.0F};
    const VkRect2D scissor{{0, 0}, extent};
    vkCmdSetViewport(frame.commandBuffer, 0U, 1U, &viewport);
    vkCmdSetScissor(frame.commandBuffer, 0U, 1U, &scissor);
    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      scenePipeline);
    vkCmdBindDescriptorSets(frame.commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout,
        0U, 1U, &frame.graphicsDescriptor, 0U, nullptr);

    drawMesh(cube, static_cast<std::uint32_t>(cubeInstances.size()),
             cameraCubeOffset);
    drawMesh(sphereNear,
             static_cast<std::uint32_t>(sphereNearInstances.size()),
             cameraNearOffset);
    drawMesh(sphereMedium,
             static_cast<std::uint32_t>(sphereMediumInstances.size()),
             cameraMediumOffset);
    drawMesh(sphereFar,
             static_cast<std::uint32_t>(sphereFarInstances.size()),
             cameraFarOffset);

    if (!resonanceInstances.empty()
        && resonancePipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          resonancePipeline);
        vkCmdBindDescriptorSets(frame.commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout,
            0U, 1U, &frame.graphicsDescriptor, 0U, nullptr);
        // Pulse fronts are few and strictly capped.  Spend a modest number of
        // extra vertices at higher quality so a room-scale wave reads as a
        // circular front rather than exposing the five-sided far LOD.
        const Mesh& pulseMesh = quality <= 0 ? sphereFar
                              : quality == 1 ? sphereMedium : sphereNear;
        drawMesh(pulseMesh,
                 static_cast<std::uint32_t>(resonanceInstances.size()),
                 resonanceOffset);
    }

    if (gpuPhysicsEnabled && activeParticles > 0U) {
        vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          particlePipeline);
        vkCmdBindDescriptorSets(frame.commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout,
            0U, 1U, &frame.graphicsDescriptor, 0U, nullptr);
        constexpr VkDeviceSize particleOffset = 0U;
        vkCmdBindVertexBuffers(frame.commandBuffer, 0U, 1U,
                               &particles.handle, &particleOffset);
        vkCmdDraw(frame.commandBuffer, activeParticles, 1U, 0U, 0U);
    }
    if (frame.hudVertexCount > 0U && hudPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          hudPipeline);
        constexpr VkDeviceSize hudOffset = 0U;
        vkCmdBindVertexBuffers(frame.commandBuffer, 0U, 1U,
                               &frame.hudVertices.handle, &hudOffset);
        vkCmdDraw(frame.commandBuffer, frame.hudVertexCount, 1U, 0U, 0U);
    }
    endRendering(frame.commandBuffer);
    if (timeFrame) {
        vkCmdWriteTimestamp(frame.commandBuffer,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            frame.timestampQueries, kTimestampGraphicsEnd);
    }

    VkImageMemoryBarrier2 presentBarrier = vkInitialize<VkImageMemoryBarrier2>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
    presentBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    presentBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    presentBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    presentBarrier.dstAccessMask = VK_ACCESS_2_NONE;
    presentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    presentBarrier.image = swapchainImages[imageIndex];
    presentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    presentBarrier.subresourceRange.levelCount = 1U;
    presentBarrier.subresourceRange.layerCount = 1U;
    VkDependencyInfo presentDependency = vkInitialize<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
    presentDependency.imageMemoryBarrierCount = 1U;
    presentDependency.pImageMemoryBarriers = &presentBarrier;
    pipelineBarrier2(frame.commandBuffer, &presentDependency);
    requireVk(vkEndCommandBuffer(frame.commandBuffer), "vkEndCommandBuffer");

    swapchainLayouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    depthImages[imageIndex].initialized = true;
}

bool Renderer::Impl::draw(const PhysicsWorld& world, float alpha,
                          float frameDeltaSeconds) {
    if (!ready || lost) {
        if (error.empty()) {
            error = lost ? "Vulkan device is lost" : "Renderer is not initialized";
        }
        return false;
    }
    if (platform == nullptr || !platform->valid()) {
        error = "Xlib window is no longer valid";
        return false;
    }
    const auto platformWidth = static_cast<std::uint32_t>(
        std::max(platform->width(), 0));
    const auto platformHeight = static_cast<std::uint32_t>(
        std::max(platform->height(), 0));
    if (platformWidth == 0U || platformHeight == 0U) {
        return true;
    }
    if (swapchain == VK_NULL_HANDLE || extent.width != platformWidth
        || extent.height != platformHeight) {
        if (!recreateSwapchain()) {
            return false;
        }
    }

    FrameContext& frame = frames[frameIndex];
    VkResult result = vkWaitForFences(
        device, 1U, &frame.fence, VK_TRUE,
        std::numeric_limits<std::uint64_t>::max());
    if (result != VK_SUCCESS) {
        lost = result == VK_ERROR_DEVICE_LOST;
        error = vkFailure("vkWaitForFences", result);
        return false;
    }
    collectFrameTimestamps(frame);
    std::uint32_t imageIndex = 0U;
    bool acquireSuboptimal = false;
    if (!acquireImage(frame, imageIndex, acquireSuboptimal)) {
        return false;
    }
    if (imageIndex >= imagesInFlight.size()) {
        error = "vkAcquireNextImageKHR returned an invalid image index";
        return false;
    }
    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE
        && imagesInFlight[imageIndex] != frame.fence) {
        result = vkWaitForFences(
            device, 1U, &imagesInFlight[imageIndex], VK_TRUE,
            std::numeric_limits<std::uint64_t>::max());
        if (result != VK_SUCCESS) {
            lost = result == VK_ERROR_DEVICE_LOST;
            error = vkFailure("vkWaitForFences(image)", result);
            return false;
        }
    }

    try {
        updateScene(world, alpha, frame);
        record(frame, imageIndex, world, frameDeltaSeconds);
    } catch (const std::exception& failure) {
        error = failure.what();
        return false;
    }
    requireVk(vkResetFences(device, 1U, &frame.fence), "vkResetFences");

    const VkSemaphoreSubmitInfo waitInfo{
        VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr,
        frame.imageAvailable, 0U, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0U};
    const VkCommandBufferSubmitInfo commandInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr,
        frame.commandBuffer, 0U};
    const VkSemaphoreSubmitInfo signalInfo{
        VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr,
        frame.renderFinished, 0U, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0U};
    VkSubmitInfo2 submit = vkInitialize<VkSubmitInfo2>(VK_STRUCTURE_TYPE_SUBMIT_INFO_2);
    submit.waitSemaphoreInfoCount = 1U;
    submit.pWaitSemaphoreInfos = &waitInfo;
    submit.commandBufferInfoCount = 1U;
    submit.pCommandBufferInfos = &commandInfo;
    submit.signalSemaphoreInfoCount = 1U;
    submit.pSignalSemaphoreInfos = &signalInfo;
    result = queueSubmit2(queue, 1U, &submit, frame.fence);
    if (result != VK_SUCCESS) {
        lost = result == VK_ERROR_DEVICE_LOST;
        error = vkFailure("vkQueueSubmit2", result);
        return false;
    }
    imagesInFlight[imageIndex] = frame.fence;
    if (timestampsSupported && frame.timestampQueries != VK_NULL_HANDLE) {
        frame.timestampPending = true;
        frame.timestampComputeActive = gpuPhysicsEnabled && activeParticles > 0U
                                    && frameDeltaSeconds > 0.0F;
        frame.timestampShadowActive = flashlightEnabled
                                   && flashlightShadowsEnabled
                                   && shadowAvailable;
    }
    if (gpuPhysicsEnabled && activeParticles > 0U
        && frameDeltaSeconds > 0.0F) {
        ++simulatedFrames;
    }
    if (flashlightEnabled && flashlightShadowsEnabled && shadowAvailable) {
        ++shadowedFrames;
    }

    VkPresentInfoKHR present = vkInitialize<VkPresentInfoKHR>(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR);
    present.waitSemaphoreCount = 1U;
    present.pWaitSemaphores = &frame.renderFinished;
    present.swapchainCount = 1U;
    present.pSwapchains = &swapchain;
    present.pImageIndices = &imageIndex;
    result = vkQueuePresentKHR(queue, &present);
    frameIndex = (frameIndex + 1U) % kFrameCount;
    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
        ++renderedFrames;
        if (acquireSuboptimal || result == VK_SUBOPTIMAL_KHR) {
            (void)recreateSwapchain();
        }
        return true;
    }
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return recreateSwapchain();
    }
    lost = result == VK_ERROR_DEVICE_LOST;
    error = vkFailure("vkQueuePresentKHR", result);
    return false;
}

VulkanStats Renderer::Impl::stats() const noexcept {
    std::uint64_t heapBytes = 0U;
    std::uint64_t budgetBytes = 0U;
    std::uint64_t usageBytes = 0U;
    if (localHeapIndex < memoryProperties.memoryHeapCount) {
        heapBytes = memoryProperties.memoryHeaps[localHeapIndex].size;
        if (memoryBudget && physicalDevice != VK_NULL_HANDLE) {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = vkInitialize<VkPhysicalDeviceMemoryBudgetPropertiesEXT>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT);
            VkPhysicalDeviceMemoryProperties2 memory2 = vkInitialize<VkPhysicalDeviceMemoryProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2);
            memory2.pNext = &budget;
            vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memory2);
            budgetBytes = budget.heapBudget[localHeapIndex];
            usageBytes = budget.heapUsage[localHeapIndex];
        }
    }
    const double timingDivisor = timedFrames > 0U
        ? static_cast<double>(timedFrames) : 1.0;
    return {
        deviceName,
        driverName,
        properties.apiVersion,
        loaderApiVersion,
        heapBytes,
        budgetBytes,
        usageBytes,
        queueFamily,
        static_cast<std::uint32_t>(swapchainImages.size()),
        renderedFrames,
        validationErrors.load(std::memory_order_relaxed),
        core13,
        ready,
        ready,
        timeline,
        memoryBudget,
        software,
        shadowAvailable,
        flashlightShadowsEnabled,
        shadowedFrames,
        timestampsSupported,
        timedFrames,
        gpuTotalMillisecondsSum / timingDivisor,
        gpuComputeMillisecondsSum / timingDivisor,
        gpuShadowMillisecondsSum / timingDivisor,
        gpuGraphicsMillisecondsSum / timingDivisor,
    };
}

void Renderer::Impl::cleanup() noexcept {
    ready = false;
    if (device != VK_NULL_HANDLE) {
        if (vkDeviceWaitIdle(device) == VK_SUCCESS) {
            collectAllFrameTimestamps();
        }
    }
    destroySwapchain();
    if (device != VK_NULL_HANDLE && shadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, shadowPipeline, nullptr);
    }
    shadowPipeline = VK_NULL_HANDLE;
    if (device != VK_NULL_HANDLE && computePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, computePipeline, nullptr);
    }
    computePipeline = VK_NULL_HANDLE;
    destroyBuffer(particles);
    destroyBuffer(sphereFar.indices);
    destroyBuffer(sphereFar.vertices);
    destroyBuffer(sphereMedium.indices);
    destroyBuffer(sphereMedium.vertices);
    destroyBuffer(sphereNear.indices);
    destroyBuffer(sphereNear.vertices);
    destroyBuffer(cube.indices);
    destroyBuffer(cube.vertices);
    sphereFar = {};
    sphereMedium = {};
    sphereNear = {};
    cube = {};
    if (device != VK_NULL_HANDLE && computePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
    }
    if (device != VK_NULL_HANDLE && graphicsPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);
    }
    computePipelineLayout = VK_NULL_HANDLE;
    graphicsPipelineLayout = VK_NULL_HANDLE;
    if (device != VK_NULL_HANDLE && descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }
    descriptorPool = VK_NULL_HANDLE;
    if (device != VK_NULL_HANDLE && materialSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, materialSampler, nullptr);
    }
    materialSampler = VK_NULL_HANDLE;
    destroyTextureArray(materialNormal);
    destroyTextureArray(materialAlbedo);
    if (device != VK_NULL_HANDLE && shadowSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, shadowSampler, nullptr);
    }
    shadowSampler = VK_NULL_HANDLE;
    if (device != VK_NULL_HANDLE && computeSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, computeSetLayout, nullptr);
    }
    if (device != VK_NULL_HANDLE && graphicsSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, graphicsSetLayout, nullptr);
    }
    computeSetLayout = VK_NULL_HANDLE;
    graphicsSetLayout = VK_NULL_HANDLE;
    for (FrameContext& frame : frames) {
        destroyBuffer(frame.hudVertices);
        destroyBuffer(frame.impacts);
        destroyBuffer(frame.instances);
        destroyBuffer(frame.uniform);
        destroyShadowMap(frame.shadowMap);
        if (device != VK_NULL_HANDLE
            && frame.timestampQueries != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, frame.timestampQueries, nullptr);
        }
        if (device != VK_NULL_HANDLE && frame.renderFinished != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, frame.renderFinished, nullptr);
        }
        if (device != VK_NULL_HANDLE && frame.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, frame.imageAvailable, nullptr);
        }
        if (device != VK_NULL_HANDLE && frame.fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, frame.fence, nullptr);
        }
        if (device != VK_NULL_HANDLE && frame.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, frame.commandPool, nullptr);
        }
        frame = {};
    }
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
    }
    device = VK_NULL_HANDLE;
    queue = VK_NULL_HANDLE;
    if (instance != VK_NULL_HANDLE && surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }
    surface = VK_NULL_HANDLE;
    if (instance != VK_NULL_HANDLE && debugMessenger != VK_NULL_HANDLE) {
        auto destroyDebug = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyDebug != nullptr) {
            destroyDebug(instance, debugMessenger, nullptr);
        }
    }
    debugMessenger = VK_NULL_HANDLE;
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
    instance = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    platform = nullptr;
    beginRendering = nullptr;
    endRendering = nullptr;
    pipelineBarrier2 = nullptr;
    queueSubmit2 = nullptr;
    cubeInstances.clear();
    sphereNearInstances.clear();
    sphereMediumInstances.clear();
    sphereFarInstances.clear();
    resonanceInstances.clear();
    shadowCubeInstances.clear();
    shadowSphereNearInstances.clear();
    shadowSphereMediumInstances.clear();
    shadowSphereFarInstances.clear();
    hudVerticesCpu.clear();
    particleCapacity = 0U;
    activeParticles = 0U;
    frameSerial = 0U;
    impactParticleCursor = 0U;
    simulatedFrames = 0U;
    gpuTime = 0.0;
    timestampValidBits = 0U;
    timestampPeriodNanoseconds = 0.0;
    gpuComputeMillisecondsSum = 0.0;
    gpuShadowMillisecondsSum = 0.0;
    gpuGraphicsMillisecondsSum = 0.0;
    gpuTotalMillisecondsSum = 0.0;
    timedFrames = 0U;
    shadowedFrames = 0U;
    shadowFormat = VK_FORMAT_UNDEFINED;
    shadowAvailable = false;
    shadowLinearFiltering = false;
    timestampsSupported = false;
    probeBody = -1;
    lastImpactSimulationTime = -1.0;
    lastResonanceSimulationTime = -1.0;
    resonancePulses.fill(ResonancePulse{});
    resonancePulseCursor = 0U;
}

Renderer::Renderer() : impl_(std::make_unique<Impl>()) {}

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::initialize(const Platform& platform, std::uint32_t seed,
                          bool maximumGpuPhysics, bool enableValidation,
                          bool allowSoftwareDevice) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    if (impl_->ready) {
        return true;
    }
    impl_->error.clear();
    impl_->lost = false;
    impl_->validationErrors.store(0U, std::memory_order_relaxed);
    impl_->maximumGpu = maximumGpuPhysics;
    impl_->platform = &platform;
    try {
        constexpr std::size_t reservePerBatch = kInstanceCapacity / 4U;
        impl_->cubeInstances.reserve(reservePerBatch);
        impl_->sphereNearInstances.reserve(reservePerBatch);
        impl_->sphereMediumInstances.reserve(reservePerBatch);
        impl_->sphereFarInstances.reserve(reservePerBatch);
        impl_->resonanceInstances.reserve(kResonancePulseCapacity);
        impl_->shadowCubeInstances.reserve(reservePerBatch);
        impl_->shadowSphereNearInstances.reserve(reservePerBatch);
        impl_->shadowSphereMediumInstances.reserve(reservePerBatch);
        impl_->shadowSphereFarInstances.reserve(reservePerBatch);
        impl_->hudVerticesCpu.reserve(kHudVertexCapacity);
        impl_->createInstance(enableValidation);
        impl_->createSurface();
        impl_->choosePhysicalDevice(allowSoftwareDevice);
        impl_->createDevice();
        impl_->loadDeviceCommands();

        constexpr std::array<VkFormat, 3> depthCandidates{{
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D24_UNORM_S8_UINT,
            VK_FORMAT_D16_UNORM,
        }};
        for (VkFormat format : depthCandidates) {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(
                impl_->physicalDevice, format, &properties);
            if ((properties.optimalTilingFeatures
                 & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0U) {
                impl_->depthFormat = format;
                break;
            }
        }
        if (impl_->depthFormat == VK_FORMAT_UNDEFINED) {
            throw std::runtime_error("No supported Vulkan depth format was found");
        }

        impl_->chooseShadowFormat();
        impl_->createFrameContexts();
        impl_->createShadowSampler();
        impl_->createMaterialTextures();
        impl_->createMaterialSampler();
        impl_->createDescriptors();
        impl_->createMeshes();
        impl_->createShadowPipeline();
        impl_->createParticles(seed);
        impl_->createComputePipeline();
        if (!impl_->createSwapchain()) {
            throw std::runtime_error("Cannot create a zero-sized Vulkan swapchain");
        }
        impl_->ready = true;
        return true;
    } catch (const std::exception& failure) {
        impl_->error = failure.what();
        impl_->cleanup();
        return false;
    }
}

void Renderer::shutdown() noexcept {
    if (impl_) {
        impl_->cleanup();
    }
}

void Renderer::waitIdle() noexcept {
    if (!impl_ || impl_->device == VK_NULL_HANDLE) {
        return;
    }
    const VkResult result = vkDeviceWaitIdle(impl_->device);
    if (result != VK_SUCCESS) {
        impl_->lost = result == VK_ERROR_DEVICE_LOST;
        impl_->error = vkFailure("vkDeviceWaitIdle", result);
    } else {
        impl_->collectAllFrameTimestamps();
    }
}

void Renderer::resetPerformanceTimings() noexcept {
    if (!impl_) {
        return;
    }
    if (impl_->device != VK_NULL_HANDLE) {
        const VkResult result = vkDeviceWaitIdle(impl_->device);
        if (result != VK_SUCCESS) {
            impl_->lost = result == VK_ERROR_DEVICE_LOST;
            impl_->error = vkFailure("vkDeviceWaitIdle(timing reset)", result);
            return;
        }
        // Every submitted query is now complete. Harvesting before clearing
        // also clears each frame's pending marker, so no pre-reset sample can
        // be collected into the next measurement window.
        impl_->collectAllFrameTimestamps();
    }
    impl_->gpuComputeMillisecondsSum = 0.0;
    impl_->gpuShadowMillisecondsSum = 0.0;
    impl_->gpuGraphicsMillisecondsSum = 0.0;
    impl_->gpuTotalMillisecondsSum = 0.0;
    impl_->timedFrames = 0U;
    impl_->shadowedFrames = 0U;
}

void Renderer::setQuality(int quality) noexcept {
    if (!impl_) {
        return;
    }
    impl_->quality = clamp(quality, 0, 2);
    if (!impl_->gpuPhysicsEnabled) {
        impl_->activeParticles = 0U;
        return;
    }
    if (impl_->maximumGpu) {
        impl_->activeParticles = impl_->particleCapacity;
        return;
    }
    constexpr std::array<std::uint32_t, 3> counts{{
        kSafeParticles, kBalancedParticles, kUltraParticles}};
    impl_->activeParticles = std::min(
        impl_->particleCapacity,
        counts[static_cast<std::size_t>(impl_->quality)]);
}

void Renderer::setMaximumGpuPhysics(bool enabled) noexcept {
    if (!impl_) {
        return;
    }
    impl_->maximumGpu = enabled;
    if (!impl_->gpuPhysicsEnabled) {
        impl_->activeParticles = 0U;
    } else if (enabled) {
        impl_->activeParticles = impl_->particleCapacity;
    } else {
        setQuality(impl_->quality);
    }
}

void Renderer::setGpuPhysicsEnabled(bool enabled) noexcept {
    if (!impl_) {
        return;
    }
    impl_->gpuPhysicsEnabled = enabled;
    if (!enabled) {
        impl_->activeParticles = 0U;
    } else if (impl_->maximumGpu) {
        impl_->activeParticles = impl_->particleCapacity;
    } else {
        setQuality(impl_->quality);
    }
}

void Renderer::setOverheadLightsEnabled(bool enabled) noexcept {
    if (impl_) {
        impl_->overheadLightsEnabled = enabled;
    }
}

void Renderer::setFlashlightEnabled(bool enabled) noexcept {
    if (impl_) {
        impl_->flashlightEnabled = enabled;
    }
}

void Renderer::setFlashlightShadowsEnabled(bool enabled) noexcept {
    if (impl_) {
        impl_->flashlightShadowsEnabled = enabled;
    }
}

void Renderer::setMaterialTexturesEnabled(bool enabled) noexcept {
    if (impl_) {
        impl_->materialTexturesEnabled = enabled;
    }
}

void Renderer::setNormalMappingEnabled(bool enabled) noexcept {
    if (impl_) {
        impl_->normalMappingEnabled = enabled;
    }
}

void Renderer::setHudState(const RendererHudState& state) noexcept {
    if (impl_) {
        impl_->hudState = state;
        impl_->hudState.quality = clamp(state.quality, 0, 2);
        impl_->hudState.fps = std::isfinite(state.fps)
            ? std::max(0.0, state.fps) : 0.0;
    }
}

bool Renderer::render(const PhysicsWorld& world, float interpolationAlpha,
                      float frameDeltaSeconds) {
    if (!impl_) {
        return false;
    }
    try {
        return impl_->draw(world, interpolationAlpha, frameDeltaSeconds);
    } catch (const std::exception& failure) {
        impl_->error = failure.what();
        return false;
    }
}

bool Renderer::initialized() const noexcept {
    return impl_ && impl_->ready;
}

bool Renderer::validationClean() const noexcept {
    return impl_
        && (!impl_->validationRequested
            || (impl_->validationEnabled
                && impl_->debugMessenger != VK_NULL_HANDLE))
        && impl_->validationErrors.load(std::memory_order_relaxed) == 0U;
}

bool Renderer::deviceLost() const noexcept {
    return impl_ && impl_->lost;
}

std::string_view Renderer::lastError() const noexcept {
    return impl_ ? std::string_view(impl_->error) : std::string_view{};
}

GpuPhysicsStats Renderer::gpuPhysicsStats() const noexcept {
    if (!impl_) {
        return {};
    }
    return {
        impl_->ready && impl_->computePipeline != VK_NULL_HANDLE
            && impl_->particles.handle != VK_NULL_HANDLE,
        impl_->activeParticles,
        impl_->particleCapacity,
        static_cast<std::uint64_t>(impl_->particles.size),
        impl_->simulatedFrames,
    };
}

VulkanStats Renderer::vulkanStats() const noexcept {
    return impl_ ? impl_->stats() : VulkanStats{};
}

} // namespace nec
