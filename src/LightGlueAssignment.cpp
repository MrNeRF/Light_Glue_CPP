#include "LightGlueModules.hpp"

MatchAssignment::MatchAssignment(int dim)
    : dim_(dim),
      matchability_(torch::nn::Linear(torch::nn::LinearOptions(dim_, 1).bias(true))), // Adjust the dimensions as needed
      final_proj_(torch::nn::LinearOptions(dim_, dim_).bias(true))                    // Adjust the dimensions as needed
{
    register_module("matchability", matchability_);
    register_module("final_proj", final_proj_);
}

torch::Tensor MatchAssignment::forward(
    const torch::Tensor& desc0,
    const torch::Tensor& desc1) {

    // Project descriptors
    auto mdesc0 = final_proj_->forward(desc0);
    auto mdesc1 = final_proj_->forward(desc1);

    // Scale by dimension
    auto d = mdesc0.size(-1);
    auto scale = 1.0f / std::pow(d, 0.25f);
    mdesc0 = mdesc0 * scale;
    mdesc1 = mdesc1 * scale;

    // Log shapes and statistics
    std::cout << "mdesc0 shape: " << mdesc0.sizes()
              << ", mean: " << mdesc0.mean().item<float>()
              << ", std: " << mdesc0.std().item<float>() << std::endl;
    std::cout << "mdesc1 shape: " << mdesc1.sizes()
              << ", mean: " << mdesc1.mean().item<float>()
              << ", std: " << mdesc1.std().item<float>() << std::endl;

    auto sim = torch::einsum("bmd,bnd->bmn", {mdesc0, mdesc1});
    std::cout << "sim shape: " << sim.sizes()
              << ", mean: " << sim.mean().item<float>()
              << ", std: " << sim.std().item<float>() << std::endl;

    auto z0 = matchability_->forward(desc0);
    auto z1 = matchability_->forward(desc1);
    std::cout << "z0 shape: " << z0.sizes()
              << ", mean: " << z0.mean().item<float>()
              << ", std: " << z0.std().item<float>() << std::endl;
    std::cout << "z1 shape: " << z1.sizes()
              << ", mean: " << z1.mean().item<float>()
              << ", std: " << z1.std().item<float>() << std::endl;

    auto scores = sigmoid_log_double_softmax(sim, z0, z1);
    std::cout << "scores shape: " << scores.sizes()
              << ", mean: " << scores.mean().item<float>()
              << ", std: " << scores.std().item<float>() << std::endl;

    return scores;
}

torch::Tensor MatchAssignment::get_matchability(const torch::Tensor& desc) {
    // Debug input tensor
    auto weight = matchability_->weight.data();
    auto bias = matchability_->bias.data();
    std::cout << "matchability weight mean: " << weight.mean().item<float>()
              << ", std: " << weight.std().item<float>() << std::endl;
    std::cout << "matchability bias mean: " << bias.mean().item<float>()
              << ", std: " << bias.std().item<float>() << std::endl;

    auto result = torch::sigmoid(matchability_->forward(desc)).squeeze(-1);

    // Debug output tensor
    std::cout << "get_matchability -> Output shape: " << result.sizes()
              << ", mean: " << result.mean().item<float>()
              << ", std: " << result.std().item<float>() << std::endl;

    return result;
}

torch::Tensor MatchAssignment::sigmoid_log_double_softmax(
    const torch::Tensor& sim,
    const torch::Tensor& z0,
    const torch::Tensor& z1) {

    auto batch_size = sim.size(0);
    auto m = sim.size(1);
    auto n = sim.size(2);

    auto certainties = torch::log_sigmoid(z0) +
                       torch::log_sigmoid(z1).transpose(1, 2);
    auto scores0 = torch::log_softmax(sim, 2);
    auto scores1 = torch::log_softmax(
                       sim.transpose(-1, -2).contiguous(), 2)
                       .transpose(-1, -2);

    auto scores = torch::full(
        {batch_size, m + 1, n + 1}, 0.0f,
        torch::TensorOptions().device(sim.device()).dtype(sim.dtype()));

    scores.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(torch::indexing::None, m),
         torch::indexing::Slice(torch::indexing::None, n)},
        scores0 + scores1 + certainties);

    scores.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(torch::indexing::None, -1),
         n},
        torch::log_sigmoid(-z0.squeeze(-1)));

    scores.index_put_(
        {torch::indexing::Slice(),
         m,
         torch::indexing::Slice(torch::indexing::None, -1)},
        torch::log_sigmoid(-z1.squeeze(-1)));

    std::cout << "sigmoid_log_double_softmax -> scores shape: " << scores.sizes()
              << ", mean: " << scores.mean().item<float>()
              << ", std: " << scores.std().item<float>() << std::endl;

    return scores;
}
