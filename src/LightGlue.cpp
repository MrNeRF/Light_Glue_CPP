#include "LightGlue.hpp"

#include "LightGlueModules.hpp"
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

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
        register_module("transformer_" + std::to_string(i), layer);
    }

    // Initialize assignment and token confidence layers
    for (int i = 0; i < config_.n_layers; ++i)
    {
        auto assign = std::make_shared<MatchAssignment>(config_.descriptor_dim);
        log_assignment_.push_back(assign);
        register_module("assignment_" + std::to_string(i), assign);

        if (i < config_.n_layers - 1)
        {
            auto conf = std::make_shared<TokenConfidence>(config_.descriptor_dim);
            token_confidence_.push_back(conf);
            register_module("confidence_" + std::to_string(i), conf);
        }
    }

    // Register confidence thresholds buffer
    std::vector<float> thresholds;
    thresholds.reserve(config_.n_layers);
    for (int i = 0; i < config_.n_layers; ++i)
    {
        thresholds.push_back(confidence_threshold(i));
    }

    register_buffer("confidence_thresholds",
                    torch::tensor(thresholds, torch::kFloat32));

    // Load weights if specified
    if (!config_.weights.empty())
    {
        load_weights(feature_type);
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

    torch::Tensor s;
    if (!size.has_value())
    {
        s = 1 + std::get<0>(torch::max(kpts, -2)) - std::get<0>(torch::min(kpts, -2));
    } else
    {
        s = size.value().to(kpts.dtype());
    }

    auto shift = s / 2;
    auto scale = std::get<0>(torch::max(s, -1)) / 2;
    return (kpts - shift.unsqueeze(-2)) / scale.unsqueeze(-2).unsqueeze(-1);
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

    // Concatenate confidences
    auto confidences = torch::cat({confidences0, confidences1}, -1);

    // Get threshold for current layer
    auto threshold = confidence_thresholds_[layer_index];

    // Calculate ratio of confident points
    auto ratio_confident = 1.0f -
                           (confidences < threshold).to(torch::kFloat32).sum().item<float>() / num_points;

    return ratio_confident > config_.depth_confidence;
}

torch::Dict<std::string, torch::Tensor> LightGlue::forward(
    const torch::Dict<std::string, torch::Tensor>& data0,
    const torch::Dict<std::string, torch::Tensor>& data1) {

    // Extract keypoints and descriptors
    auto kpts0 = data0.at("keypoints");
    auto kpts1 = data1.at("keypoints");
    auto desc0 = data0.at("descriptors").detach().contiguous();
    auto desc1 = data1.at("descriptors").detach().contiguous();

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

    // Normalize keypoints
    kpts0 = normalize_keypoints(kpts0, size0).clone();
    kpts1 = normalize_keypoints(kpts1, size1).clone();

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

    // Convert to fp16 if mixed precision is enabled
    if (config_.mp && device_.is_cuda())
    {
        desc0 = desc0.to(torch::kHalf);
        desc1 = desc1.to(torch::kHalf);
    }

    // Initialize masks and handle padding for compilation
    torch::optional<torch::Tensor> mask0, mask1;
    int64_t c = std::max(m, n);
    bool do_compile = !static_lengths_.empty() && c <= *std::max_element(static_lengths_.begin(), static_lengths_.end());

    if (do_compile)
    {
        auto kn = *std::lower_bound(static_lengths_.begin(), static_lengths_.end(), c);
        std::tie(desc0, mask0) = pad_to_length(desc0, kn);
        std::tie(desc1, mask1) = pad_to_length(desc1, kn);
        // auto dummy_mask = torch::ones({1}, torch::TensorOptions().dtype(torch::kBool).device(device_));
        std::tie(kpts0, std::ignore) = pad_to_length(kpts0, kn);
        std::tie(kpts1, std::ignore) = pad_to_length(kpts1, kn);
    }

    // Project descriptors if needed
    if (config_.input_dim != config_.descriptor_dim)
    {
        desc0 = input_proj_->forward(desc0);
        desc1 = input_proj_->forward(desc1);
    }

    // Generate positional encodings
    auto encoding0 = posenc_->forward(kpts0);
    auto encoding1 = posenc_->forward(kpts1);

    // Initialize pruning if enabled
    bool do_early_stop = config_.depth_confidence > 0;
    bool do_point_pruning = config_.width_confidence > 0 && !do_compile;
    auto pruning_th = pruning_keypoint_thresholds_.at(
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
        std::tie(desc0, desc1) = transformers_[i]->forward(
            desc0, desc1, encoding0, encoding1, mask0, mask1);

        if (i == config_.n_layers - 1)
            continue;

        // Early stopping check
        if (do_early_stop)
        {
            std::tie(token0, token1) = token_confidence_[i]->forward(desc0, desc1);
            if (check_if_stop(
                    token0.value().index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, m)}),
                    token1.value().index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, n)}),
                    i, m + n))
            {
                break;
            }
        }

        // Point pruning
        if (do_point_pruning && desc0.size(-2) > pruning_th)
        {
            auto scores0 = log_assignment_[i]->get_matchability(desc0);
            auto prunemask0 = get_pruning_mask(token0, scores0, i);
            auto where_result = torch::where(prunemask0);
            auto keep0 = where_result[1];
            ind0 = ind0.index_select(1, keep0);
            desc0 = desc0.index_select(1, keep0);
            encoding0 = encoding0.index_select(-2, keep0);
            prune0.index_put_({torch::indexing::Slice(), ind0}, prune0.index({torch::indexing::Slice(), ind0}) + 1);
        }

        if (do_point_pruning && desc1.size(-2) > pruning_th)
        {
            auto scores1 = log_assignment_[i]->get_matchability(desc1);
            auto prunemask1 = get_pruning_mask(token1, scores1, i);
            auto where_result = torch::where(prunemask1);
            const auto& keep1 = where_result[1];
            ind1 = ind1.index_select(1, keep1);
            desc1 = desc1.index_select(1, keep1);
            encoding1 = encoding1.index_select(-2, keep1);
            prune1.index_put_({torch::indexing::Slice(), ind1}, prune1.index({torch::indexing::Slice(), ind1}) + 1);
        }
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

    auto [scores, sim] = log_assignment_[i]->forward(desc0, desc1);
    auto [m0, m1, mscores0, mscores1] = filter_matches(scores, config_.filter_threshold);

    // Update matches if point pruning was used
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
    output.insert("stop", torch::tensor(i + 1));
    output.insert("prune0", prune0);
    output.insert("prune1", prune1);

    return output;
}

void LightGlue::load_weights(const std::string& feature_type) {
    auto model_path = "../models/" + config_.weights + ".pt";

    if (!fs::exists(model_path))
    {
        throw std::runtime_error("Cannot find pretrained model: " + model_path);
    }

    std::cout << "Loading " << model_path << std::endl;
    load_parameters(model_path);
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
        const auto name = w.key().toStringRef();
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