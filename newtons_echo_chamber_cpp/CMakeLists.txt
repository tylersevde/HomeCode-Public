cmake_minimum_required(VERSION 3.20)

project(newtons_echo_chamber_alpha LANGUAGES CXX)

include(CTest)
include(CheckIPOSupported)
include(GNUInstallDirs)

option(NEC_NATIVE_TUNING
       "Tune release builds for the Raspberry Pi 5 Cortex-A76"
       ON)
option(NEC_ENABLE_LTO
       "Enable interprocedural optimization in release builds"
       ON)

find_package(Vulkan 1.3 REQUIRED)
find_package(X11 REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

# FindVulkan exposes this variable when glslc is installed alongside the Vulkan
# SDK. Raspberry Pi OS packages it separately, so retain a PATH fallback.
if(NOT Vulkan_GLSLC_EXECUTABLE)
    find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc REQUIRED)
endif()

set(NEC_SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(GLOB NEC_SHADER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp"
)

set(NEC_SHADER_BINARIES)
foreach(NEC_SHADER_SOURCE IN LISTS NEC_SHADER_SOURCES)
    get_filename_component(NEC_SHADER_NAME "${NEC_SHADER_SOURCE}" NAME)
    set(NEC_SHADER_BINARY
        "${NEC_SHADER_BINARY_DIR}/${NEC_SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT "${NEC_SHADER_BINARY}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${NEC_SHADER_BINARY_DIR}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}"
                --target-env=vulkan1.2
                -o "${NEC_SHADER_BINARY}"
                "${NEC_SHADER_SOURCE}"
        DEPENDS "${NEC_SHADER_SOURCE}"
        COMMENT "Compiling Vulkan shader ${NEC_SHADER_NAME}"
        VERBATIM
    )
    list(APPEND NEC_SHADER_BINARIES "${NEC_SHADER_BINARY}")
endforeach()

add_custom_target(newtons_vulkan_shaders ALL DEPENDS ${NEC_SHADER_BINARIES})

set(NEC_TEXTURE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/material_albedo.ppm")
set(NEC_TEXTURE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/textures")
set(NEC_TEXTURE_ALBEDO
    "${NEC_TEXTURE_BINARY_DIR}/material_albedo.ppm")
set(NEC_TEXTURE_NORMAL
    "${NEC_TEXTURE_BINARY_DIR}/material_normal.ppm")

add_executable(newtons_texture_builder tools/texture_builder.cpp)
target_compile_features(newtons_texture_builder PRIVATE cxx_std_20)
set_target_properties(newtons_texture_builder PROPERTIES CXX_EXTENSIONS OFF)

add_custom_command(
    OUTPUT "${NEC_TEXTURE_ALBEDO}" "${NEC_TEXTURE_NORMAL}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NEC_TEXTURE_BINARY_DIR}"
    COMMAND $<TARGET_FILE:newtons_texture_builder>
            "${NEC_TEXTURE_SOURCE}"
            "${NEC_TEXTURE_ALBEDO}"
            "${NEC_TEXTURE_NORMAL}"
    DEPENDS newtons_texture_builder "${NEC_TEXTURE_SOURCE}"
    COMMENT "Building wrap-safe albedo and tangent-space normal atlases"
    VERBATIM
)
add_custom_target(newtons_texture_assets ALL
    DEPENDS "${NEC_TEXTURE_ALBEDO}" "${NEC_TEXTURE_NORMAL}")

add_executable(newtons_echo_chamber_alpha
    src/main.cpp
    src/physics.cpp
    src/renderer.cpp
    src/platform.cpp
    src/audio.cpp
)
add_dependencies(newtons_echo_chamber_alpha
    newtons_vulkan_shaders
    newtons_texture_assets)

target_compile_features(newtons_echo_chamber_alpha PRIVATE cxx_std_20)
set_target_properties(newtons_echo_chamber_alpha PROPERTIES
    CXX_EXTENSIONS OFF
)
target_compile_definitions(newtons_echo_chamber_alpha PRIVATE
    _POSIX_C_SOURCE=200809L
    VK_USE_PLATFORM_XLIB_KHR=1
    NEC_SHADER_BINARY_DIR="${NEC_SHADER_BINARY_DIR}"
    NEC_INSTALL_SHADER_DIR="${CMAKE_INSTALL_FULL_DATADIR}/newtons_echo_chamber/shaders"
    NEC_TEXTURE_ASSET_DIR="${NEC_TEXTURE_BINARY_DIR}"
    NEC_INSTALL_TEXTURE_DIR="${CMAKE_INSTALL_FULL_DATADIR}/newtons_echo_chamber/textures"
)
target_include_directories(newtons_echo_chamber_alpha PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(newtons_echo_chamber_alpha PRIVATE
    Vulkan::Vulkan
    X11::X11
    Threads::Threads
    ZLIB::ZLIB
    ${CMAKE_DL_LIBS}
    m
)

add_executable(newtons_vulkan_probe tools/vulkan_probe.cpp)
target_compile_features(newtons_vulkan_probe PRIVATE cxx_std_20)
set_target_properties(newtons_vulkan_probe PROPERTIES
    CXX_EXTENSIONS OFF
)
target_link_libraries(newtons_vulkan_probe PRIVATE Vulkan::Vulkan)

add_executable(newtons_gpu_physics_probe
    tools/gpu_physics_probe.cpp
    src/physics.cpp
)
add_dependencies(newtons_gpu_physics_probe newtons_vulkan_shaders)
target_compile_features(newtons_gpu_physics_probe PRIVATE cxx_std_20)
set_target_properties(newtons_gpu_physics_probe PROPERTIES
    CXX_EXTENSIONS OFF
)
target_compile_definitions(newtons_gpu_physics_probe PRIVATE
    _POSIX_C_SOURCE=200809L
    NEC_SHADER_BINARY_DIR="${NEC_SHADER_BINARY_DIR}"
    NEC_INSTALL_SHADER_DIR="${CMAKE_INSTALL_FULL_DATADIR}/newtons_echo_chamber/shaders"
)
target_include_directories(newtons_gpu_physics_probe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(newtons_gpu_physics_probe PRIVATE
    Vulkan::Vulkan
    Threads::Threads
    m
)

function(nec_enable_warnings TARGET_NAME)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET_NAME} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wformat=2
        )
    endif()
endfunction()

nec_enable_warnings(newtons_echo_chamber_alpha)
nec_enable_warnings(newtons_vulkan_probe)
nec_enable_warnings(newtons_gpu_physics_probe)
nec_enable_warnings(newtons_texture_builder)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(newtons_echo_chamber_alpha PRIVATE
        $<$<CONFIG:Release>:-fno-math-errno>
        $<$<CONFIG:RelWithDebInfo>:-fno-math-errno>
        $<$<CONFIG:Release>:-fno-trapping-math>
        $<$<CONFIG:RelWithDebInfo>:-fno-trapping-math>
        $<$<CONFIG:Release>:-ffp-contract=fast>
        $<$<CONFIG:RelWithDebInfo>:-ffp-contract=fast>
    )

    if(NEC_NATIVE_TUNING AND
       CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        foreach(NEC_TARGET IN ITEMS
                newtons_echo_chamber_alpha
                newtons_vulkan_probe
                newtons_gpu_physics_probe)
            target_compile_options(${NEC_TARGET} PRIVATE
                $<$<CONFIG:Release>:-mcpu=cortex-a76>
                $<$<CONFIG:RelWithDebInfo>:-mcpu=cortex-a76>
                $<$<CONFIG:Release>:-mtune=cortex-a76>
                $<$<CONFIG:RelWithDebInfo>:-mtune=cortex-a76>
            )
        endforeach()
    endif()
endif()

if(NEC_ENABLE_LTO)
    check_ipo_supported(RESULT NEC_IPO_SUPPORTED OUTPUT NEC_IPO_ERROR)
    if(NEC_IPO_SUPPORTED)
        foreach(NEC_TARGET IN ITEMS
                newtons_echo_chamber_alpha
                newtons_vulkan_probe
                newtons_gpu_physics_probe)
            set_property(TARGET ${NEC_TARGET} PROPERTY
                INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
            set_property(TARGET ${NEC_TARGET} PROPERTY
                INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
        endforeach()
    else()
        message(WARNING "LTO is unavailable: ${NEC_IPO_ERROR}")
    endif()
endif()

if(BUILD_TESTING)
    add_test(
        NAME newtons_deterministic_check
        COMMAND newtons_echo_chamber_alpha --check
    )
    set_tests_properties(newtons_deterministic_check PROPERTIES TIMEOUT 30)
    add_test(
        NAME newtons_texture_asset_check
        COMMAND newtons_texture_builder --check "${NEC_TEXTURE_SOURCE}"
    )
    set_tests_properties(newtons_texture_asset_check PROPERTIES TIMEOUT 30)
endif()

install(TARGETS
    newtons_echo_chamber_alpha
    newtons_vulkan_probe
    newtons_gpu_physics_probe
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(FILES ${NEC_SHADER_BINARIES}
    DESTINATION "${CMAKE_INSTALL_DATADIR}/newtons_echo_chamber/shaders"
)
install(FILES "${NEC_TEXTURE_ALBEDO}" "${NEC_TEXTURE_NORMAL}"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/newtons_echo_chamber/textures"
)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/LICENSE.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/MANIFEST.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/CC0_SOURCES.md"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/newtons_echo_chamber/textures"
)
