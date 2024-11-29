#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <memory>

// Forward declarations
class LearnableFourierPosEnc;
class TokenConfidence;
class TransformerLayer;
class MatchAssignment;

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

class LightGlue : public torch::nn::Module {
public:
    explicit LightGlue(const std::string& feature_type = "aliked",
                       const LightGlueConfig& config = LightGlueConfig());

    // Main forward function to process features and find matches
    torch::Dict<std::string, torch::Tensor> forward(
        const torch::Dict<std::string, torch::Tensor>& data0,
        const torch::Dict<std::string, torch::Tensor>& data1);

    // Method to move all components to specified device
    void to(const torch::Device& device);

private:
    // Helper functions
    static torch::Tensor normalize_keypoints(const torch::Tensor& kpts,
                                             const torch::optional<torch::Tensor>& size = torch::nullopt);

    std::tuple<torch::Tensor, torch::Tensor> pad_to_length(
        const torch::Tensor& x, int64_t length);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    filter_matches(const torch::Tensor& scores, float threshold);

    float confidence_threshold(int layer_index) const;

    torch::Tensor get_pruning_mask(
        const torch::optional<torch::Tensor>& confidences,
        const torch::Tensor& scores,
        int layer_index);

    bool check_if_stop(
        const torch::Tensor& confidences0,
        const torch::Tensor& confidences1,
        int layer_index,
        int num_points);

    void load_weights(const std::string& feature_type);

private:
    LightGlueConfig config_;
    torch::Device device_;

    // Neural network components
    torch::nn::Linear input_proj_{nullptr};
    std::shared_ptr<LearnableFourierPosEnc> posenc_;
    std::vector<std::shared_ptr<TransformerLayer>> transformers_;
    std::vector<std::shared_ptr<MatchAssignment>> log_assignment_;
    std::vector<std::shared_ptr<TokenConfidence>> token_confidence_;
    std::vector<float> confidence_thresholds_;

    static const std::unordered_map<std::string, int> pruning_keypoint_thresholds_;
    void load_parameters(const std::string& pt_path);
    std::vector<char> get_the_bytes(const std::string& filename);
};