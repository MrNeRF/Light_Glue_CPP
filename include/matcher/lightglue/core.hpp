#pragma once
#include <string>
#include <torch/torch.h>

namespace matcher {
    struct LightGlueConfig {
        std::string name = "lightglue";
        int input_dim = 128;
        int descriptor_dim = 256;
        bool add_scale_ori = false;
        int n_layers = 9;
        int num_heads = 4;
        bool flash = false;
        bool mp = false;
        float depth_confidence = 0.95f;
        float width_confidence = 0.99f;
        float filter_threshold = 0.1f;
        std::string weights;
    };
}

namespace matcher::utils {
    torch::Tensor normalize_keypoints(const torch::Tensor& kpts,
                                             const torch::optional<torch::Tensor>& size = torch::nullopt);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    filter_matches(const torch::Tensor& scores, float threshold);
}