#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kAtlasSize = 1024U;
constexpr std::size_t kTileSize = 256U;
constexpr std::size_t kTilesPerAxis = 4U;
constexpr std::size_t kChannelCount = 3U;
constexpr std::size_t kPixelBytes = kAtlasSize * kAtlasSize * kChannelCount;
constexpr std::size_t kSeamBlendWidth = 24U;

struct Image {
    std::size_t width{};
    std::size_t height{};
    std::vector<std::uint8_t> pixels;
};

[[nodiscard]] bool isWhitespace(char value) noexcept {
    return value == ' ' || value == '\t' || value == '\n'
        || value == '\r' || value == '\f' || value == '\v';
}

void skipWhitespaceAndComments(std::istream& input) {
    while (true) {
        const int next = input.peek();
        if (next == std::char_traits<char>::eof()) {
            throw std::runtime_error("unexpected end of PPM header");
        }
        if (isWhitespace(static_cast<char>(next))) {
            static_cast<void>(input.get());
            continue;
        }
        if (next != '#') {
            return;
        }
        static_cast<void>(input.get());
        while (true) {
            const int comment = input.get();
            if (comment == std::char_traits<char>::eof()) {
                throw std::runtime_error("unterminated PPM comment");
            }
            if (comment == '\n' || comment == '\r') {
                break;
            }
        }
    }
}

[[nodiscard]] std::string readToken(std::istream& input) {
    skipWhitespaceAndComments(input);
    std::string token;
    while (true) {
        const int next = input.peek();
        if (next == std::char_traits<char>::eof()
            || isWhitespace(static_cast<char>(next)) || next == '#') {
            break;
        }
        token.push_back(static_cast<char>(input.get()));
        if (token.size() > 32U) {
            throw std::runtime_error("unreasonably long PPM header token");
        }
    }
    if (token.empty()) {
        throw std::runtime_error("missing PPM header token");
    }
    return token;
}

[[nodiscard]] std::size_t parseUnsigned(std::string_view text,
                                        std::string_view label) {
    std::size_t value = 0U;
    for (const char character : text) {
        if (character < '0' || character > '9') {
            throw std::runtime_error(std::string(label) + " is not an integer");
        }
        const std::size_t digit = static_cast<std::size_t>(character - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
            throw std::runtime_error(std::string(label) + " overflows");
        }
        value = value * 10U + digit;
    }
    return value;
}

[[nodiscard]] Image readPpm(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open input texture: " + path.string());
    }
    if (readToken(input) != "P6") {
        throw std::runtime_error("input must be a binary P6 PPM");
    }
    const std::size_t width = parseUnsigned(readToken(input), "PPM width");
    const std::size_t height = parseUnsigned(readToken(input), "PPM height");
    const std::size_t maximum = parseUnsigned(readToken(input), "PPM maxval");
    if (width != kAtlasSize || height != kAtlasSize) {
        throw std::runtime_error("texture atlas must be exactly 1024x1024 pixels");
    }
    if (maximum != 255U) {
        throw std::runtime_error("texture atlas maxval must be exactly 255");
    }

    const int separator = input.get();
    if (separator == std::char_traits<char>::eof()
        || !isWhitespace(static_cast<char>(separator))) {
        throw std::runtime_error("PPM header must end with whitespace");
    }
    // A CRLF header is one separator, not a leading 0x0a image byte.
    if (separator == '\r' && input.peek() == '\n') {
        static_cast<void>(input.get());
    }

    Image image{width, height, std::vector<std::uint8_t>(kPixelBytes)};
    input.read(reinterpret_cast<char*>(image.pixels.data()),
               static_cast<std::streamsize>(image.pixels.size()));
    if (input.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
        throw std::runtime_error("PPM pixel payload is truncated");
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("PPM contains trailing data after its pixel payload");
    }
    return image;
}

[[nodiscard]] std::size_t pixelOffset(std::size_t x, std::size_t y) noexcept {
    return (y * kAtlasSize + x) * kChannelCount;
}

[[nodiscard]] double smoothStep(double value) noexcept {
    const double t = std::clamp(value, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

[[nodiscard]] std::uint8_t blendChannel(std::uint8_t original,
                                        unsigned average,
                                        double originalWeight) noexcept {
    const double value = static_cast<double>(average) * (1.0 - originalWeight)
                       + static_cast<double>(original) * originalWeight;
    return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

[[nodiscard]] Image makeWrapSafeAlbedo(const Image& source) {
    Image result = source;
    for (std::size_t tileY = 0U; tileY < kTilesPerAxis; ++tileY) {
        for (std::size_t tileX = 0U; tileX < kTilesPerAxis; ++tileX) {
            const std::size_t originX = tileX * kTileSize;
            const std::size_t originY = tileY * kTileSize;
            for (std::size_t distance = 0U; distance < kSeamBlendWidth;
                 ++distance) {
                const double weight = smoothStep(static_cast<double>(distance)
                    / static_cast<double>(kSeamBlendWidth - 1U));
                const std::size_t leftX = originX + distance;
                const std::size_t rightX = originX + kTileSize - 1U - distance;
                for (std::size_t y = originY; y < originY + kTileSize; ++y) {
                    const std::size_t left = pixelOffset(leftX, y);
                    const std::size_t right = pixelOffset(rightX, y);
                    for (std::size_t channel = 0U; channel < kChannelCount;
                         ++channel) {
                        const auto average = (static_cast<unsigned>(result.pixels[left + channel])
                            + static_cast<unsigned>(result.pixels[right + channel]) + 1U) / 2U;
                        const std::uint8_t leftValue = result.pixels[left + channel];
                        const std::uint8_t rightValue = result.pixels[right + channel];
                        result.pixels[left + channel] = blendChannel(
                            leftValue, average, weight);
                        result.pixels[right + channel] = blendChannel(
                            rightValue, average, weight);
                    }
                }
            }

            for (std::size_t distance = 0U; distance < kSeamBlendWidth;
                 ++distance) {
                const double weight = smoothStep(static_cast<double>(distance)
                    / static_cast<double>(kSeamBlendWidth - 1U));
                const std::size_t topY = originY + distance;
                const std::size_t bottomY = originY + kTileSize - 1U - distance;
                for (std::size_t x = originX; x < originX + kTileSize; ++x) {
                    const std::size_t top = pixelOffset(x, topY);
                    const std::size_t bottom = pixelOffset(x, bottomY);
                    for (std::size_t channel = 0U; channel < kChannelCount;
                         ++channel) {
                        const auto average = (static_cast<unsigned>(result.pixels[top + channel])
                            + static_cast<unsigned>(result.pixels[bottom + channel]) + 1U) / 2U;
                        const std::uint8_t topValue = result.pixels[top + channel];
                        const std::uint8_t bottomValue = result.pixels[bottom + channel];
                        result.pixels[top + channel] = blendChannel(
                            topValue, average, weight);
                        result.pixels[bottom + channel] = blendChannel(
                            bottomValue, average, weight);
                    }
                }
            }
        }
    }
    return result;
}

[[nodiscard]] float luminance(const Image& image, std::size_t x,
                              std::size_t y) noexcept {
    const std::size_t offset = pixelOffset(x, y);
    return (0.2126F * static_cast<float>(image.pixels[offset])
          + 0.7152F * static_cast<float>(image.pixels[offset + 1U])
          + 0.0722F * static_cast<float>(image.pixels[offset + 2U])) / 255.0F;
}

[[nodiscard]] std::size_t wrappedCoordinate(std::size_t origin,
                                            int coordinate) noexcept {
    constexpr int size = static_cast<int>(kTileSize);
    int wrapped = coordinate % size;
    if (wrapped < 0) {
        wrapped += size;
    }
    return origin + static_cast<std::size_t>(wrapped);
}

[[nodiscard]] std::uint8_t encodeNormal(float value) noexcept {
    const long encoded = std::lround((value * 0.5F + 0.5F) * 255.0F);
    return static_cast<std::uint8_t>(std::clamp(encoded, 0L, 255L));
}

[[nodiscard]] Image makeNormalAtlas(const Image& albedo) {
    Image normal{albedo.width, albedo.height,
                 std::vector<std::uint8_t>(kPixelBytes)};
    constexpr float strength = 3.25F;
    for (std::size_t tileY = 0U; tileY < kTilesPerAxis; ++tileY) {
        for (std::size_t tileX = 0U; tileX < kTilesPerAxis; ++tileX) {
            const std::size_t originX = tileX * kTileSize;
            const std::size_t originY = tileY * kTileSize;
            const auto height = [&](int x, int y) noexcept {
                return luminance(albedo, wrappedCoordinate(originX, x),
                                 wrappedCoordinate(originY, y));
            };
            for (int y = 0; y < static_cast<int>(kTileSize); ++y) {
                for (int x = 0; x < static_cast<int>(kTileSize); ++x) {
                    const float gradientX =
                        (height(x + 1, y - 1) + 2.0F * height(x + 1, y)
                         + height(x + 1, y + 1)
                         - height(x - 1, y - 1) - 2.0F * height(x - 1, y)
                         - height(x - 1, y + 1)) * 0.25F;
                    const float gradientY =
                        (height(x - 1, y + 1) + 2.0F * height(x, y + 1)
                         + height(x + 1, y + 1)
                         - height(x - 1, y - 1) - 2.0F * height(x, y - 1)
                         - height(x + 1, y - 1)) * 0.25F;
                    float nx = -gradientX * strength;
                    float ny = -gradientY * strength;
                    float nz = 1.0F;
                    const float inverseLength = 1.0F
                        / std::sqrt(nx * nx + ny * ny + nz * nz);
                    nx *= inverseLength;
                    ny *= inverseLength;
                    nz *= inverseLength;
                    const std::size_t offset = pixelOffset(
                        originX + static_cast<std::size_t>(x),
                        originY + static_cast<std::size_t>(y));
                    normal.pixels[offset] = encodeNormal(nx);
                    normal.pixels[offset + 1U] = encodeNormal(ny);
                    normal.pixels[offset + 2U] = encodeNormal(nz);
                }
            }
        }
    }
    return normal;
}

[[nodiscard]] double channelVariance(const Image& image, std::size_t tile,
                                     bool normalMap) noexcept {
    const std::size_t originX = (tile % kTilesPerAxis) * kTileSize;
    const std::size_t originY = (tile / kTilesPerAxis) * kTileSize;
    double mean = 0.0;
    double squared = 0.0;
    std::size_t samples = 0U;
    for (std::size_t y = originY; y < originY + kTileSize; ++y) {
        for (std::size_t x = originX; x < originX + kTileSize; ++x) {
            const std::size_t offset = pixelOffset(x, y);
            const double value = normalMap
                ? static_cast<double>(image.pixels[offset])
                    + 0.7071067811865476 * static_cast<double>(image.pixels[offset + 1U])
                : 0.2126 * static_cast<double>(image.pixels[offset])
                    + 0.7152 * static_cast<double>(image.pixels[offset + 1U])
                    + 0.0722 * static_cast<double>(image.pixels[offset + 2U]);
            ++samples;
            const double delta = value - mean;
            mean += delta / static_cast<double>(samples);
            squared += delta * (value - mean);
        }
    }
    return squared / static_cast<double>(samples);
}

void verifyAtlas(const Image& albedo, const Image& normal) {
    for (std::size_t tile = 0U; tile < kTilesPerAxis * kTilesPerAxis; ++tile) {
        const double albedoVariance = channelVariance(albedo, tile, false);
        const double normalVariance = channelVariance(normal, tile, true);
        if (!(albedoVariance > 0.01)) {
            throw std::runtime_error("albedo tile " + std::to_string(tile)
                                     + " is flat");
        }
        if (!(normalVariance > 0.01)) {
            throw std::runtime_error("normal tile " + std::to_string(tile)
                                     + " is flat");
        }
    }

    // The outer rows/columns of each tile are equal after seam processing.
    // This lets a renderer clamp atlas coordinates to texel centers and repeat
    // local UVs without introducing a hard material seam.
    for (std::size_t tileY = 0U; tileY < kTilesPerAxis; ++tileY) {
        for (std::size_t tileX = 0U; tileX < kTilesPerAxis; ++tileX) {
            const std::size_t originX = tileX * kTileSize;
            const std::size_t originY = tileY * kTileSize;
            for (std::size_t offset = 0U; offset < kTileSize; ++offset) {
                for (std::size_t channel = 0U; channel < kChannelCount; ++channel) {
                    const auto left = albedo.pixels[pixelOffset(originX,
                        originY + offset) + channel];
                    const auto right = albedo.pixels[pixelOffset(
                        originX + kTileSize - 1U, originY + offset) + channel];
                    const auto top = albedo.pixels[pixelOffset(
                        originX + offset, originY) + channel];
                    const auto bottom = albedo.pixels[pixelOffset(
                        originX + offset, originY + kTileSize - 1U) + channel];
                    if (left != right || top != bottom) {
                        throw std::runtime_error("seam processing failed for tile "
                                                 + std::to_string(tileY * kTilesPerAxis + tileX));
                    }
                }
            }
        }
    }
}

void writePpm(const std::filesystem::path& path, const Image& image) {
    std::error_code directoryError;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), directoryError);
        if (directoryError) {
            throw std::runtime_error("cannot create output directory for "
                                     + path.string() + ": " + directoryError.message());
        }
    }
    const std::filesystem::path temporary = path.string() + ".tmp";
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot open output texture: " + temporary.string());
        }
        output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
        output.write(reinterpret_cast<const char*>(image.pixels.data()),
                     static_cast<std::streamsize>(image.pixels.size()));
        output.close();
        if (!output) {
            throw std::runtime_error("failed while writing texture: " + temporary.string());
        }
        std::error_code renameError;
        std::filesystem::rename(temporary, path, renameError);
        if (renameError) {
            std::filesystem::remove(path, renameError);
            renameError.clear();
            std::filesystem::rename(temporary, path, renameError);
        }
        if (renameError) {
            throw std::runtime_error("cannot commit output texture " + path.string()
                                     + ": " + renameError.message());
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void printUsage(std::ostream& output, std::string_view program) {
    output << "Usage:\n  " << program << " --check INPUT.ppm\n  "
           << program << " INPUT.ppm OUTPUT_ALBEDO.ppm OUTPUT_NORMAL.ppm\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool checkOnly = false;
        std::filesystem::path inputPath;
        std::filesystem::path albedoPath;
        std::filesystem::path normalPath;
        if (argc == 3 && std::string_view(argv[1]) == "--check") {
            checkOnly = true;
            inputPath = argv[2];
        } else if (argc == 4) {
            inputPath = argv[1];
            albedoPath = argv[2];
            normalPath = argv[3];
        } else {
            printUsage(std::cerr, argc > 0 ? argv[0] : "newtons_texture_builder");
            return 2;
        }

        const Image source = readPpm(inputPath);
        const Image albedo = makeWrapSafeAlbedo(source);
        const Image normal = makeNormalAtlas(albedo);
        verifyAtlas(albedo, normal);
        if (!checkOnly) {
            writePpm(albedoPath, albedo);
            writePpm(normalPath, normal);
        }
        std::cout << (checkOnly ? "texture-check" : "texture-build")
                  << ": 16/16 variable wrap-safe 256x256 material tiles; "
                  << "normal atlas derived from periodic luminance gradients\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "texture-builder: " << error.what() << '\n';
        return 1;
    }
}
