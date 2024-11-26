# ALIKED C++ Implementation

This is a C++ implementation of ALIKED (Attentive Local and Implicit Keypoint Detector) using LibTorch and OpenCV. The implementation provides a high-performance, production-ready version of the ALIKED model for keypoint detection and matching.
For more, join my discord server: https://discord.com/invite/NqwTqVYVmj 

## Features

- Complete C++ implementation of ALIKED model
- CUDA-accelerated computations
- OpenCV integration for image processing
- Real-time keypoint detection and matching
- Multiple model configurations (aliked-t16, aliked-n16, aliked-n16rot, aliked-n32)
- Move semantics optimization for better performance
- Simple tracking demo application

## Prerequisites

- CMake (>= 3.26)
- CUDA Toolkit (>= 12.1)
- LibTorch (with CUDA support)
- OpenCV
- C++20 compatible compiler

## Directory Structure

```
.
├── include/               # Header files
├── src/                  # Source files
├── examples/             # Example applications
├── models/               # Pre-trained model weights
├── external/            
│   └── libtorch/        # LibTorch directory
└── CMakeLists.txt       # CMake configuration
```

## Setup Instructions

1. Download and extract LibTorch:
   ```bash
   mkdir -p external
   cd external
   wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
   cd ..
   ```

2. Build the project:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j$(nproc)
   ```

## Usage

The demo application performs keypoint detection and matching between consecutive images:

```bash
./aliked /path/to/image/directory [options]
```

### Options

- `model_name`: Model configuration (default: "aliked-n32")
- `device`: Computation device (default: "cuda")
- `top_k`: Number of top keypoints (-1 for threshold-based selection, default: -1)
- `scores_th`: Score threshold for keypoint selection (default: 0.2)
- `n_limit`: Maximum number of keypoints (default: 5000)

### Example Code

```cpp
#include "ALIKED.hpp"

// Initialize model
auto model = std::make_shared<ALIKED>("aliked-n32", "cuda");

// Load and process image
cv::Mat img = cv::imread("image.jpg");
cv::Mat img_rgb;
cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

// Run inference
auto pred = model->run(img_rgb);
auto keypoints = pred.at("keypoints");
auto descriptors = pred.at("descriptors");
```

## Model Configurations

| Model Name    | Description                               |
|--------------|-------------------------------------------|
| aliked-t16   | Tiny model with 16 descriptor dimensions  |
| aliked-n16   | Normal model with 16 descriptor dimensions|
| aliked-n16rot| Rotation-invariant model                  |
| aliked-n32   | Normal model with 32 descriptor dimensions|

## Performance Optimizations

The implementation includes several optimizations:

- Link Time Optimization (LTO/IPO)
- CPU architecture-specific optimizations
- CUDA optimizations
- Fast math operations
- Position-independent code

## Custom Model Directory

You can specify a custom location for model weights during build:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DALIKED_MODELS_DIR=/path/to/models ..
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

## Acknowledgements

- Original ALIKED paper and implementation
- LibTorch and PyTorch teams
- OpenCV team