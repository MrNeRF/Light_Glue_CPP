#include "LightGlue.hpp"

#include "LightGlueModules.hpp"
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

std::string map_python_to_cpp(const std::string& python_name) {
    std::string cpp_name = python_name;

    // Replace "transformers" with "transformer"
    size_t pos_transformer = cpp_name.find("transformers");
    size_t pos_assignment = cpp_name.find("log_assignment");
    size_t pos_confidence = cpp_name.find("token_confidence");

    size_t pos = std::min({pos_transformer, pos_assignment, pos_confidence});
    // Replace ".<digit>" with "<digit>"
    size_t dot_pos = cpp_name.find_first_of("0123456789", pos);
    if (dot_pos != std::string::npos && cpp_name[dot_pos - 1] == '.')
    {
        cpp_name.erase(dot_pos - 1, 1); // Remove the dot before the number
    }

    return cpp_name;
}

// Static member initialization
const std::unordered_map<std::string, int> LightGlue::pruning_keypoint_thresholds_ = {
    {"cpu", -1},
    {"mps", -1},
    {"cuda", 1024},
    {"flash", 1536}};

// Feature configurations
static const std::unordered_map<std::string, std::pair<std::string, int>> FEATURES = {
    {"aliked", {"aliked_lightglue", 128}}};

LightGlue::LightGlue(const std::string& feature_type, const LightGlueConfig& config)
    : config_(config),
      device_(torch::kCPU) {

    // Configure based on feature type
    auto it = FEATURES.find(feature_type);
    if (it == FEATURES.end())
    {
        throw std::runtime_error("Unsupported feature type: " + feature_type);
    }

    config_.weights = it->second.first;
    config_.input_dim = it->second.second;

    // Initialize input projection if needed
    if (config_.input_dim != config_.descriptor_dim)
    {
        input_proj_ = register_module("input_proj",
                                      torch::nn::Linear(config_.input_dim, config_.descriptor_dim));
    }

    // Initialize positional encoding
    posenc_ = register_module("posenc",
                              std::make_shared<LearnableFourierPosEnc>(
                                  2 + 2 * config_.add_scale_ori,
                                  config_.descriptor_dim / config_.num_heads,
                                  config_.descriptor_dim / config_.num_heads));

    // Initialize transformer layers
    for (int i = 0; i < config_.n_layers; ++i)
    {
        auto layer = std::make_shared<TransformerLayer>(
            config_.descriptor_dim,
            config_.num_heads,
            config_.flash);

        transformers_.push_back(layer);
        register_module("transformers" + std::to_string(i), layer);
    }

    // Initialize assignment and token confidence layers
    for (int i = 0; i < config_.n_layers; ++i)
    {
        auto assign = std::make_shared<MatchAssignment>(config_.descriptor_dim);
        log_assignment_.push_back(assign);
        register_module("log_assignment" + std::to_string(i), assign);

        if (i < config_.n_layers - 1)
        {
            auto conf = std::make_shared<TokenConfidence>(config_.descriptor_dim);
            token_confidence_.push_back(conf);
            register_module("token_confidence" + std::to_string(i), conf);
        }
    }

    // Register confidence thresholds buffer
    confidence_thresholds_.reserve(config_.n_layers);
    for (int i = 0; i < config_.n_layers; ++i)
    {
        confidence_thresholds_.push_back(confidence_threshold(i));
    }

    // Load weights if specified
    if (!config_.weights.empty())
    {
        load_weights(config_.weights);
    }

    // Move to device if CUDA is available
    if (torch::cuda::is_available())
    {
        device_ = torch::kCUDA;
        this->to(device_);
    }
}

void LightGlue::to(const torch::Device& device) {
    device_ = device;
    torch::nn::Module::to(device);
}

float LightGlue::confidence_threshold(int layer_index) const {
    float progress = static_cast<float>(layer_index) / config_.n_layers;
    float threshold = 0.8f + 0.1f * std::exp(-4.0f * progress);
    return std::clamp(threshold, 0.0f, 1.0f);
}

torch::Tensor LightGlue::normalize_keypoints(
    const torch::Tensor& kpts,
    const torch::optional<torch::Tensor>& size) {

    torch::Tensor size_tensor;
    if (!size.has_value())
    {
        // Compute the size as the range of keypoints
        size_tensor = 1 + std::get<0>(torch::max(kpts, /*dim=*/-2)) - std::get<0>(torch::min(kpts, /*dim=*/-2));
    } else
    {
        // If size is provided but not a tensor, convert it to a tensor
        size_tensor = size.value().to(kpts);
    }

    // Compute shift and scale
    auto shift = size_tensor / 2;
    auto scale = std::get<0>(size_tensor.max(-1)) / 2;

    return (kpts - shift.unsqueeze(-2)) / scale.unsqueeze(-1).unsqueeze(-1);
}

std::tuple<torch::Tensor, torch::Tensor> LightGlue::pad_to_length(
    const torch::Tensor& x,
    int64_t length) {

    if (length <= x.size(-2))
    {
        return std::make_tuple(
            x,
            torch::ones({*x.sizes().begin(), x.size(-2), 1},
                        torch::TensorOptions()
                            .dtype(torch::kBool)
                            .device(x.device())));
    }

    auto pad = torch::ones(
        {*x.sizes().begin(), length - x.size(-2), x.size(-1)},
        torch::TensorOptions()
            .dtype(x.dtype())
            .device(x.device()));

    auto y = torch::cat({x, pad}, -2);
    auto mask = torch::zeros(
        {*y.sizes().begin(), y.size(-2), 1},
        torch::TensorOptions()
            .dtype(torch::kBool)
            .device(x.device()));

    mask.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(torch::indexing::None, x.size(-2)),
         torch::indexing::Slice()},
        true);

    return std::make_tuple(y, mask);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LightGlue::filter_matches(const torch::Tensor& scores, float threshold) {
    // Get max values along both dimensions
    auto max0 = std::get<1>(scores.slice(1, 0, -1).slice(2, 0, -1).max(2));
    auto max1 = std::get<1>(scores.slice(1, 0, -1).slice(2, 0, -1).max(1));

    // Create index tensors
    auto indices0 = torch::arange(max0.size(1),
                                  torch::TensorOptions().device(max0.device()))
                        .unsqueeze(0);
    auto indices1 = torch::arange(max1.size(1),
                                  torch::TensorOptions().device(max1.device()))
                        .unsqueeze(0);

    // Check mutual matches
    auto mutual0 = indices0 == max1.gather(1, max0);
    auto mutual1 = indices1 == max0.gather(1, max1);

    // Calculate match scores
    auto max0_exp = std::get<0>(scores.slice(1, 0, -1)
                                    .slice(2, 0, -1)
                                    .max(2))
                        .exp();

    auto zero = torch::zeros_like(max0_exp);
    auto mscores0 = torch::where(mutual0, max0_exp, zero);
    auto mscores1 = torch::where(mutual1, mscores0.gather(1, max1), zero);

    // Apply threshold
    auto valid0 = mutual0 & (mscores0 > threshold);
    auto valid1 = mutual1 & valid0.gather(1, max1);

    // Create match indices
    auto m0 = torch::where(valid0, max0,
                           torch::full_like(max0, -1, max0.dtype()));
    auto m1 = torch::where(valid1, max1,
                           torch::full_like(max1, -1, max1.dtype()));

    return std::make_tuple(m0, m1, mscores0, mscores1);
}

torch::Tensor LightGlue::get_pruning_mask(
    const torch::optional<torch::Tensor>& confidences,
    const torch::Tensor& scores,
    int layer_index) {

    // Initialize keep mask based on scores
    auto keep = scores > (1.0f - config_.width_confidence);

    // Include low-confidence points if confidences are provided
    if (confidences.has_value())
    {
        keep = keep | (confidences.value() <= confidence_thresholds_[layer_index]);
    }

    return keep;
}

bool LightGlue::check_if_stop(
    const torch::Tensor& confidences0,
    const torch::Tensor& confidences1,
    int layer_index,
    int num_points) {

    std::cout << "confidences0: " << confidences0.sizes() << std::endl;
    std::cout << "confidences1: " << confidences1.sizes() << std::endl;
    std::cout << "layer_index: " << layer_index << std::endl;
    std::cout << "num_points: " << num_points << std::endl;
    // Concatenate confidences
    auto confidences = torch::cat({confidences0, confidences1}, -1);
    std::cout << "confidences: " << confidences.sizes() << std::endl;

    // Get threshold for current layer
    auto threshold = confidence_thresholds_[layer_index];
    std::cout << "threshold: " << threshold << std::endl;

    // Calculate ratio of confident points
    auto ratio_confident = 1.0f -
                           (confidences < threshold).to(torch::kFloat32).sum().item<float>() / num_points;
    std::cout << "ratio_confident: " << ratio_confident << std::endl;

    return ratio_confident > config_.depth_confidence;
}

torch::Dict<std::string, torch::Tensor> LightGlue::forward(
    const torch::Dict<std::string, torch::Tensor>& data0,
    const torch::Dict<std::string, torch::Tensor>& data1) {

    // Extract keypoints and descriptors
    // TODO: Batching
    auto kpts0 = data0.at("keypoints").unsqueeze(0);
    auto kpts1 = data1.at("keypoints").unsqueeze(0);
    auto desc0 = data0.at("descriptors").detach().contiguous().unsqueeze(0);
    auto desc1 = data1.at("descriptors").detach().contiguous().unsqueeze(0);

    std::cout << "desc0 shape: " << desc0.sizes()
              << ", mean: " << desc0.mean().item<float>()
              << ", std: " << desc0.std().item<float>() << std::endl;
    std::cout << "desc1 shape: " << desc1.sizes()
              << ", mean: " << desc1.mean().item<float>()
              << ", std: " << desc1.std().item<float>() << std::endl;

    std::cout << "kpts0 shape: " << kpts0.sizes()
              << ", mean: " << kpts0.mean().item<float>()
              << ", std: " << kpts0.std().item<float>() << std::endl;

    std::cout << "kpts1 shape: " << kpts1.sizes()
              << ", mean: " << kpts1.mean().item<float>()
              << ", std: " << kpts1.std().item<float>() << std::endl;

    // Get batch size and point counts
    int64_t b = kpts0.size(0);
    int64_t m = kpts0.size(1);
    int64_t n = kpts1.size(1);

    // Get image sizes if available
    torch::optional<torch::Tensor> size0, size1;
    if (data0.contains("image_size"))
        size0 = data0.at("image_size");
    if (data1.contains("image_size"))
        size1 = data1.at("image_size");

    std::cout << "kpts0 shape: " << kpts0.sizes()
              << ", mean: " << kpts0.mean().item<float>()
              << ", std: " << kpts0.std().item<float>() << std::endl;

    std::cout << "kpts1 shape: " << kpts1.sizes()
              << ", mean: " << kpts1.mean().item<float>()
              << ", std: " << kpts1.std().item<float>() << std::endl;
    // Normalize keypoints
    kpts0 = normalize_keypoints(kpts0, size0).clone();
    kpts1 = normalize_keypoints(kpts1, size1).clone();

    std::cout << "kpts0 shape: " << kpts0.sizes()
              << ", mean: " << kpts0.mean().item<float>()
              << ", std: " << kpts0.std().item<float>() << std::endl;

    std::cout << "kpts1 shape: " << kpts1.sizes()
              << ", mean: " << kpts1.mean().item<float>()
              << ", std: " << kpts1.std().item<float>() << std::endl;

    // Add scale and orientation if configured
    if (config_.add_scale_ori)
    {
        kpts0 = torch::cat({kpts0,
                            data0.at("scales").unsqueeze(-1),
                            data0.at("oris").unsqueeze(-1)},
                           -1);
        kpts1 = torch::cat({kpts1,
                            data1.at("scales").unsqueeze(-1),
                            data1.at("oris").unsqueeze(-1)},
                           -1);
    }

    std::cout << "kpts0 shape: " << kpts0.sizes()
              << ", mean: " << kpts0.mean().item<float>()
              << ", std: " << kpts0.std().item<float>() << std::endl;

    std::cout << "kpts1 shape: " << kpts1.sizes()
              << ", mean: " << kpts1.mean().item<float>()
              << ", std: " << kpts1.std().item<float>() << std::endl;
    // Convert to fp16 if mixed precision is enabled
    if (config_.mp && device_.is_cuda())
    {
        desc0 = desc0.to(torch::kHalf);
        desc1 = desc1.to(torch::kHalf);
    }

    // Project descriptors if needed
    if (config_.input_dim != config_.descriptor_dim)
    {
        desc0 = input_proj_->forward(desc0);
        desc1 = input_proj_->forward(desc1);
    }
    std::cout << "desc0 shape: " << desc0.sizes()
              << ", mean: " << desc0.mean().item<float>()
              << ", std: " << desc0.std().item<float>() << std::endl;
    std::cout << "desc1 shape: " << desc1.sizes()
              << ", mean: " << desc1.mean().item<float>()
              << ", std: " << desc1.std().item<float>() << std::endl;

    // Generate positional encodings
    auto encoding0 = posenc_->forward(kpts0);
    auto encoding1 = posenc_->forward(kpts1);
    std::cout << "encoding0 shape: " << encoding0.sizes()
              << ", mean: " << encoding0.mean().item<float>()
              << ", std: " << encoding0.std().item<float>() << std::endl;
    std::cout << "encoding1 shape: " << encoding1.sizes()
              << ", mean: " << encoding1.mean().item<float>()
              << ", std: " << encoding1.std().item<float>() << std::endl;

    // Initialize pruning if enabled
    const bool do_early_stop = config_.depth_confidence > 0.f;
    const bool do_point_pruning = config_.width_confidence > 0.f;
    const auto pruning_th = pruning_keypoint_thresholds_.at(
        config_.flash ? "flash" : device_.is_cuda() ? "cuda"
                                                    : "cpu");

    torch::Tensor ind0, ind1, prune0, prune1;
    if (do_point_pruning)
    {
        ind0 = torch::arange(m, torch::TensorOptions().device(device_)).unsqueeze(0);
        ind1 = torch::arange(n, torch::TensorOptions().device(device_)).unsqueeze(0);
        prune0 = torch::ones_like(ind0);
        prune1 = torch::ones_like(ind1);
    }

    // Process through transformer layers
    torch::optional<torch::Tensor> token0, token1;
    int i;
    for (i = 0; i < config_.n_layers; ++i)
    {
        if (desc0.size(1) == 0 || desc1.size(1) == 0)
            break;

        // Process through transformer layer
        std::cout << "desc0: " << desc0.sizes() << std::endl;
        std::cout << "encoding0: " << encoding0.sizes() << std::endl;
        std::cout << "desc1: " << desc1.sizes() << std::endl;
        std::cout << "encoding1: " << encoding1.sizes() << std::endl;

        std::tie(desc0, desc1) = transformers_[i]->forward(
            desc0, desc1, encoding0, encoding1);

        if (i == config_.n_layers - 1)
            continue;

        // Early stopping check
        if (do_early_stop)
        {
            std::cout << "desc0: " << desc0.sizes() << std::endl;
            std::cout << "desc1: " << desc1.sizes() << std::endl;
            std::tie(token0, token1) = token_confidence_[i]->forward(desc0, desc1);

            std::cout << "token0: " << token0.value().sizes() << std::endl;
            std::cout << "token1: " << token1.value().sizes() << std::endl;
            if (check_if_stop(
                    token0.value().index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, m)}),
                    token1.value().index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, n)}),
                    i, m + n))
            {
                break;
            }
        }

        std::cout << "Before pruning:" << std::endl;
        std::cout << "desc0 shape: " << desc0.sizes()
                  << ", mean: " << desc0.mean().item<float>()
                  << ", std: " << desc0.std().item<float>() << std::endl;
        std::cout << "desc1 shape: " << desc1.sizes()
                  << ", mean: " << desc1.mean().item<float>()
                  << ", std: " << desc1.std().item<float>() << std::endl;

        if (do_point_pruning && desc0.size(-2) > pruning_th)
        {
            auto scores0 = log_assignment_[i]->get_matchability(desc0);
            std::cout << "scores0 shape: " << scores0.sizes()
                      << ", mean: " << scores0.mean().item<float>()
                      << ", std: " << scores0.std().item<float>() << std::endl;

            auto prunemask0 = get_pruning_mask(token0, scores0, i);
            std::cout << "prunemask0 shape: " << prunemask0.sizes()
                      << ", num_true: " << prunemask0.sum().item<int>()
                      << ", mean: " << prunemask0.to(torch::kFloat32).mean().item<float>() << std::endl;

            if (prunemask0.dtype() != torch::kBool)
            {
                prunemask0 = prunemask0.to(torch::kBool);
            }

            auto where_result = torch::where(prunemask0);
            auto keep0 = where_result[1];
            std::cout << "keep0 indices: " << keep0.sizes()
                      << ", num_kept: " << keep0.numel() << std::endl;

            if (keep0.numel() > 0)
            {
                ind0 = ind0.index_select(1, keep0);
                std::cout << "ind0 after index_select: " << ind0.sizes() << std::endl;

                desc0 = desc0.index_select(1, keep0);
                std::cout << "desc0 after index_select: " << desc0.sizes()
                          << ", mean: " << desc0.mean().item<float>()
                          << ", std: " << desc0.std().item<float>() << std::endl;

                encoding0 = encoding0.index_select(-2, keep0);
                std::cout << "encoding0 after index_select: " << encoding0.sizes()
                          << ", mean: " << encoding0.mean().item<float>()
                          << ", std: " << encoding0.std().item<float>() << std::endl;

                prune0.index_put_({torch::indexing::Slice(), ind0}, prune0.index({torch::indexing::Slice(), ind0}) + 1);
                std::cout << "prune0 after update: " << prune0.sizes() << std::endl;
            } else
            {
                std::cout << "No points kept after pruning for desc0." << std::endl;
            }
        }

        if (do_point_pruning && desc1.size(-2) > pruning_th)
        {
            auto scores1 = log_assignment_[i]->get_matchability(desc1);
            std::cout << "scores1 shape: " << scores1.sizes()
                      << ", mean: " << scores1.mean().item<float>()
                      << ", std: " << scores1.std().item<float>() << std::endl;

            auto prunemask1 = get_pruning_mask(token1, scores1, i);
            std::cout << "prunemask1 shape: " << prunemask1.sizes()
                      << ", num_true: " << prunemask1.sum().item<int>()
                      << ", mean: " << prunemask1.to(torch::kFloat32).mean().item<float>() << std::endl;

            if (prunemask1.dtype() != torch::kBool)
            {
                prunemask1 = prunemask1.to(torch::kBool);
            }

            auto where_result = torch::where(prunemask1);
            auto keep1 = where_result[1];
            std::cout << "keep1 indices: " << keep1.sizes()
                      << ", num_kept: " << keep1.numel() << std::endl;

            if (keep1.numel() > 0)
            {
                ind1 = ind1.index_select(1, keep1);
                std::cout << "ind1 after index_select: " << ind1.sizes() << std::endl;

                desc1 = desc1.index_select(1, keep1);
                std::cout << "desc1 after index_select: " << desc1.sizes()
                          << ", mean: " << desc1.mean().item<float>()
                          << ", std: " << desc1.std().item<float>() << std::endl;

                encoding1 = encoding1.index_select(-2, keep1);
                std::cout << "encoding1 after index_select: " << encoding1.sizes()
                          << ", mean: " << encoding1.mean().item<float>()
                          << ", std: " << encoding1.std().item<float>() << std::endl;

                prune1.index_put_({torch::indexing::Slice(), ind1}, prune1.index({torch::indexing::Slice(), ind1}) + 1);
                std::cout << "prune1 after update: " << prune1.sizes() << std::endl;
            } else
            {
                std::cout << "No points kept after pruning for desc1." << std::endl;
            }
        }

        std::cout << "After pruning:" << std::endl;
        std::cout << "desc0 shape: " << desc0.sizes()
                  << ", mean: " << desc0.mean().item<float>()
                  << ", std: " << desc0.std().item<float>() << std::endl;
        std::cout << "desc1 shape: " << desc1.sizes()
                  << ", mean: " << desc1.mean().item<float>()
                  << ", std: " << desc1.std().item<float>() << std::endl;
    }

    // Handle empty descriptor case
    if (desc0.size(1) == 0 || desc1.size(1) == 0)
    {
        auto m0 = torch::full({b, m}, -1, torch::TensorOptions().dtype(torch::kLong).device(device_));
        auto m1 = torch::full({b, n}, -1, torch::TensorOptions().dtype(torch::kLong).device(device_));
        auto mscores0 = torch::zeros({b, m}, device_);
        auto mscores1 = torch::zeros({b, n}, device_);

        if (!do_point_pruning)
        {
            prune0 = torch::ones_like(mscores0) * config_.n_layers;
            prune1 = torch::ones_like(mscores1) * config_.n_layers;
        }

        torch::Dict<std::string, torch::Tensor> output;
        output.insert("matches0", m0);
        output.insert("matches1", m1);
        output.insert("matching_scores0", mscores0);
        output.insert("matching_scores1", mscores1);
        output.insert("stop", torch::tensor(i + 1));
        output.insert("prune0", prune0);
        output.insert("prune1", prune1);

        return output;
    }

    // Remove padding and compute assignment
    desc0 = desc0.index({torch::indexing::Slice(),
                         torch::indexing::Slice(torch::indexing::None, m)});
    desc1 = desc1.index({torch::indexing::Slice(),
                         torch::indexing::Slice(torch::indexing::None, n)});

    std::cout << "desc0 shape: " << desc0.sizes()
              << ", mean: " << desc0.mean().item<float>()
              << ", std: " << desc0.std().item<float>() << std::endl;
    std::cout << "desc1 shape: " << desc1.sizes()
              << ", mean: " << desc1.mean().item<float>()
              << ", std: " << desc1.std().item<float>() << std::endl;
    auto scores = log_assignment_[i]->forward(desc0, desc1);
    std::cout << "scores shape: " << scores.sizes() << ", mean: " << scores.mean().item<float>() << ", std: " << scores.std().item<float>() << std::endl;

    auto [m0, m1, mscores0, mscores1] = filter_matches(scores, config_.filter_threshold);

    torch::Tensor m_indices_0, m_indices_1;

    // Create matches and scores tensors
    if (do_point_pruning)
    {
        m_indices_0 = torch::arange(b, device_).unsqueeze(1).repeat({1, m}).flatten();
        m_indices_1 = m0.flatten();

        auto valid = m_indices_1 >= 0;
        m_indices_0 = m_indices_0.index({valid});
        m_indices_1 = m_indices_1.index({valid});

        m_indices_0 = ind0.index({torch::indexing::Slice(), m_indices_0});
        m_indices_1 = ind1.index({torch::indexing::Slice(), m_indices_1});
    }

    auto matches = torch::stack({m_indices_0, m_indices_1}, 0);

    // Update m0, m1, mscores tensors
    if (do_point_pruning)
    {
        auto m0_ = torch::full({b, m}, -1, torch::TensorOptions().dtype(torch::kLong).device(device_));
        auto m1_ = torch::full({b, n}, -1, torch::TensorOptions().dtype(torch::kLong).device(device_));
        auto mscores0_ = torch::zeros({b, m}, device_);
        auto mscores1_ = torch::zeros({b, n}, device_);

        m0_.index_put_({torch::indexing::Slice(), ind0},
                       torch::where(m0 == -1, -1, ind1.gather(1, m0.clamp(0))));
        m1_.index_put_({torch::indexing::Slice(), ind1},
                       torch::where(m1 == -1, -1, ind0.gather(1, m1.clamp(0))));

        mscores0_.index_put_({torch::indexing::Slice(), ind0}, mscores0);
        mscores1_.index_put_({torch::indexing::Slice(), ind1}, mscores1);

        m0 = m0_;
        m1 = m1_;
        mscores0 = mscores0_;
        mscores1 = mscores1_;
    } else
    {
        prune0 = torch::ones_like(mscores0) * config_.n_layers;
        prune1 = torch::ones_like(mscores1) * config_.n_layers;
    }

    // Prepare output
    torch::Dict<std::string, torch::Tensor> output;
    output.insert("matches0", m0);
    output.insert("matches1", m1);
    output.insert("matching_scores0", mscores0);
    output.insert("matching_scores1", mscores1);
    output.insert("matches", matches);
    output.insert("stop", torch::tensor(i + 1));
    output.insert("prune0", prune0);
    output.insert("prune1", prune1);

    return output;
}

void LightGlue::load_weights(const std::string& feature_type) {
    std::vector<std::filesystem::path> search_paths = {
        std::filesystem::path(LIGHTGLUE_MODELS_DIR) / (std::string(feature_type) + ".pt"),
        std::filesystem::current_path() / "models" / (std::string(feature_type) + ".pt"),
        std::filesystem::current_path() / (std::string(feature_type) + ".pt")};

    std::filesystem::path model_path;
    bool found = false;

    for (const auto& path : search_paths)
    {
        if (std::filesystem::exists(path))
        {
            model_path = path;
            found = true;
            break;
        }
    }

    if (!found)
    {
        std::string error_msg = "Cannot find pretrained model. Searched in:\n";
        for (const auto& path : search_paths)
        {
            error_msg += "  " + path.string() + "\n";
        }
        error_msg += "Please place the model file in one of these locations.";
        throw std::runtime_error(error_msg);
    }

    std::cout << "Loading model from: " << model_path << std::endl;
    load_parameters(model_path.string());
}

void LightGlue::load_parameters(const std::string& pt_path) {
    auto f = get_the_bytes(pt_path);
    auto weights = torch::pickle_load(f).toGenericDict();

    // Use unordered_maps for O(1) lookup
    std::unordered_map<std::string, torch::Tensor> param_map;
    std::unordered_map<std::string, torch::Tensor> buffer_map;

    auto model_params = named_parameters();
    auto model_buffers = named_buffers();
    // Pre-allocate with expected size
    param_map.reserve(model_params.size());
    buffer_map.reserve(model_buffers.size());

    // Collect parameter names
    for (const auto& p : model_params)
    {
        param_map.emplace(p.key(), p.value());
    }

    // Collect buffer names
    for (const auto& b : model_buffers)
    {
        buffer_map.emplace(b.key(), b.value());
    }

    // Update parameters and buffers
    torch::NoGradGuard no_grad;

    for (const auto& w : weights)
    {
        const auto name = map_python_to_cpp(w.key().toStringRef());
        const auto& param = w.value().toTensor();

        // Try parameters first
        if (auto it = param_map.find(name); it != param_map.end())
        {
            if (it->second.sizes() == param.sizes())
            {
                it->second.copy_(param);
            } else
            {
                throw std::runtime_error(
                    "Shape mismatch for parameter: " + name +
                    " Expected: " + std::to_string(it->second.numel()) +
                    " Got: " + std::to_string(param.numel()));
            }
            continue;
        }

        // Then try buffers
        if (auto it = buffer_map.find(name); it != buffer_map.end())
        {
            if (it->second.sizes() == param.sizes())
            {
                it->second.copy_(param);
            } else
            {
                std::cout << "buffer name: " << name << "Expected: " << it->second.sizes() << ", Got: " << param.sizes() << std::endl;
                throw std::runtime_error(
                    "Shape mismatch for buffer: " + name +
                    " Expected: " + std::to_string(it->second.numel()) +
                    " Got: " + std::to_string(param.numel()));
            }
            continue;
        }

        // Parameter not found in model
        std::cerr << "Warning: " << name
                  << " not found in model parameters or buffers\n";
    }
}

std::vector<char> LightGlue::get_the_bytes(const std::string& filename) {
    // Use RAII file handling
    std::ifstream file(std::string(filename), std::ios::binary);
    if (!file)
    {
        throw std::runtime_error(
            "Failed to open file: " + std::string(filename));
    }

    // Get file size
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Pre-allocate vector
    std::vector<char> buffer;
    buffer.reserve(size);

    // Read file in chunks for better performance
    constexpr size_t CHUNK_SIZE = 8192;
    char chunk[CHUNK_SIZE];

    while (file.read(chunk, CHUNK_SIZE))
    {
        buffer.insert(buffer.end(), chunk, chunk + file.gcount());
    }
    if (file.gcount() > 0)
    {
        buffer.insert(buffer.end(), chunk, chunk + file.gcount());
    }

    return buffer;
}