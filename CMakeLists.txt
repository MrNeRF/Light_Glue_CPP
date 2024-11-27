cmake_minimum_required(VERSION 3.26)
project(LightGlue
        VERSION 1.0.0
        DESCRIPTION "C++ implementation of LightGlue"
        LANGUAGES CUDA CXX)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

# Enable LTO/IPO only for Release builds
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Core configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure paths
set(LIGHTGLUE_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models" CACHE PATH "Path to model weights directory")

# Find dependencies
set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch")
set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Check CUDA version
if(CUDAToolkit_VERSION VERSION_LESS "12.1")
    message(FATAL_ERROR "This project requires CUDA 12.1 or higher (found: ${CUDAToolkit_VERSION})")
endif()

# Debug flags for different compilers - removed problematic flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DEBUG_FLAGS
            -g
            -Wall
            -Wextra
            -fno-omit-frame-pointer
    )
endif()

# Release flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(RELEASE_FLAGS
            -O3
            -march=native
            -mtune=native
            -fomit-frame-pointer
            -ffast-math
            -DNDEBUG
    )
endif()

# CUDA debug flags - simplified
set(CUDA_DEBUG_FLAGS
        -G
        -g
        -lineinfo
)

# CUDA release flags
set(CUDA_RELEASE_FLAGS
        -O3
        --use_fast_math
        -DNDEBUG
)

# Add models directory definition
add_definitions(-DLIGHTGLUE_MODELS_DIR="${LIGHTGLUE_MODELS_DIR}")

# Source files
set(LIGHTGLUE_HEADERS
        include/ALIKED.hpp
        include/DKD.hpp
        include/SDDH.hpp
        include/blocks.hpp
	    include/LightGlue.hpp
        include/LightGlueModules.hpp
        include/get_patches.hpp
        include/input_padder.hpp
        include/deform_conv2d.h
        include/get_patches_cuda.h
        include/cuda_helpers.h
)

set(LIGHTGLUE_SOURCES
        src/LightGlue.cpp
        src/LightGlueAssignment.cpp
        src/LightGlueBlocks.cpp
        src/LightGlueModules.cpp
        src/LightGlueTransformer.cpp
        src/blocks.cpp
        src/ALIKED.cpp
        src/DKD.cpp
        src/input_padder.cpp
        src/get_patches.cpp
        src/SDDH.cpp
        src/deform_conv2d.cpp
        src/deform_conv2d_kernel.cu
        src/get_patches_cuda.cu
)

# Library target
add_library(${PROJECT_NAME}_lib STATIC
        ${LIGHTGLUE_SOURCES}
        ${LIGHTGLUE_HEADERS}
)

add_library(${PROJECT_NAME}::lib ALIAS ${PROJECT_NAME}_lib)

target_include_directories(${PROJECT_NAME}_lib
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Configure compile options based on build type - simplified
target_compile_options(${PROJECT_NAME}_lib
        PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:${DEBUG_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:${RELEASE_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${CUDA_DEBUG_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:${CUDA_RELEASE_FLAGS}>
)

target_link_libraries(${PROJECT_NAME}_lib
        PUBLIC
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        PRIVATE
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
)

# Properties for debug/release configurations
set_target_properties(${PROJECT_NAME}_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
        DEBUG_POSTFIX "d"
)

# Debug configuration specific settings - removed problematic definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME}_lib
            PRIVATE
            DEBUG
    )
endif()

# Example application
add_executable(${PROJECT_NAME} examples/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}::lib)

# Output directories with debug/release suffixes
set_target_properties(${PROJECT_NAME} ${PROJECT_NAME}_lib PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/release"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib/debug"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib/release"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib/debug"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib/release"
)

# Create models directory if it doesn't exist
add_custom_target(create_models_dir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${LIGHTGLUE_MODELS_DIR}
)

# Print configuration information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Models directory: ${LIGHTGLUE_MODELS_DIR}")