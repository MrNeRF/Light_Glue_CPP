# LightGlue C++ Implementation

This repository contains a C++ implementation of [LightGlue](https://github.com/cvg/LightGlue) using LibTorch. LightGlue is a lightweight feature matcher that achieves state-of-the-art performance while being significantly faster than traditional approaches.

## Features

- Complete C++ implementation of LightGlue using LibTorch
- CUDA acceleration support
- Integration with OpenCV for image handling
- [ALIKED](https://github.com/MrNeRF/ALIKED_CPP) feature extractor implementation
- Efficient memory management with move semantics
- Visualization support for matches and pruning

## Prerequisites

- CMake >= 3.26
- CUDA >= 12.1
- LibTorch (C++ distribution of PyTorch)
- OpenCV
- C++20 compliant compiler

## Building

```bash
mkdir build && cd build
cmake ..
make -j
```

## Usage

The repository includes a sample application demonstrating feature matching between two images:

```bash
./LightGlue path/to/image1.jpg path/to/image2.jpg
```

## Project Structure

```
.
├── include/
│   ├── feature/         # Feature extraction components
│   └── matcher/         # LightGlue matcher implementation
├── src/
│   ├── feature/         # Feature extraction implementations
│   └── matcher/         # Matcher implementations
├── examples/            # Example applications
├── models/             # Directory for model weights
└── CMakeLists.txt
```

## Implementation Details

The implementation follows the original Python architecture while leveraging C++ and LibTorch features:
- CUDA optimizations for performance
- Move semantics for efficient memory handling
- LibTorch's automatic differentiation (though primarily used for inference)
- OpenCV integration for image processing and visualization

## Model Weights

Place the model weights in the `models/` directory. The following models are supported:
- ALIKED feature extractor weights
- LightGlue matcher weights

## Future Development

### TODO
- [ ] Batch Processing Support
   - Implement efficient batch processing for multiple image pairs
   - Optimize memory usage for batch operations
   - Add batch-specific configuration options

- [ ] Flash Attention Implementation
   - Add efficient Flash Attention mechanism
   - Optimize for different GPU architectures
   - Implement memory-efficient attention patterns

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## Citations

If you use this implementation, please cite both the original LightGlue paper and the C++ implementation:

```bibtex
@inproceedings{lindenberger2023lightglue,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  title     = {{LightGlue: Local Feature Matching at Light Speed}},
  booktitle = {ICCV},
  year      = {2023}
}

@misc{patas2024lightgluecpp,
  author    = {Janusch Patas},
  title     = {LightGlue C++ Implementation},
  year      = {2024},
  publisher = {GitHub},
  journal   = {GitHub Repository},
  howpublished = {\url{https://github.com/MrNeRF/Light_Glue_CPP}}
}
```