cmake_minimum_required(VERSION 3.26)
project(aliked
        VERSION 1.0.0
        DESCRIPTION "C++ implementation of ALIKED"
        LANGUAGES CUDA CXX)

# Enable LTO/IPO
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Core configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configure paths
set(ALIKED_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models" CACHE PATH "Path to model weights directory")

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

# Performance flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(PERFORMANCE_FLAGS
            -O3
            -march=native
            -mtune=native
            -fomit-frame-pointer
            -ffast-math
    )
endif()

# CUDA flags
set(CUDA_FLAGS
        -O3
        --use_fast_math
        -DNDEBUG
)

# Add models directory definition
add_definitions(-DALIKED_MODELS_DIR="${ALIKED_MODELS_DIR}")

# Source files
set(ALIKED_HEADERS
        include/ALIKED.hpp
        include/DKD.hpp
        include/SDDH.hpp
        include/blocks.hpp
        include/get_patches.hpp
        include/input_padder.hpp
        include/deform_conv2d.h
        include/get_patches_cuda.h
        include/cuda_helpers.h
)

set(ALIKED_SOURCES
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
        ${ALIKED_SOURCES}
        ${ALIKED_HEADERS}
)

add_library(${PROJECT_NAME}::lib ALIAS ${PROJECT_NAME}_lib)

target_include_directories(${PROJECT_NAME}_lib
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(${PROJECT_NAME}_lib
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${PERFORMANCE_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
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

# Properties for maximum performance
set_target_properties(${PROJECT_NAME}_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
        INTERPROCEDURAL_OPTIMIZATION ${IPO_SUPPORTED}
)

# Example application
add_executable(${PROJECT_NAME} examples/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}::lib)

# Set output directories
set_target_properties(${PROJECT_NAME} ${PROJECT_NAME}_lib PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Create models directory if it doesn't exist
add_custom_target(create_models_dir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ALIKED_MODELS_DIR}
)

# Print models directory location
message(STATUS "Models directory: ${ALIKED_MODELS_DIR}")