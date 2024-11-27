#include "LightGlueModules.hpp"

MatchAssignment::MatchAssignment(int dim)
    : dim_(dim),
      matchability_(torch::nn::Linear(torch::nn::LinearOptions(dim_, 1).bias(true))), // Adjust the dimensions as needed
      final_proj_(torch::nn::LinearOptions(dim_, dim_).bias(true))  // Adjust the dimensions as needed
{
    register_module("matchability", matchability_);
    register_module("final_proj", final_proj_);
}

std::tuple<torch::Tensor, torch::Tensor> MatchAssignment::forward(
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

    // Compute similarity matrix
    auto sim = torch::einsum("bmd,bnd->bmn", {mdesc0, mdesc1});

    // Get matchability scores
    auto z0 = matchability_->forward(desc0);
    auto z1 = matchability_->forward(desc1);

    // Compute the assignment matrix
    auto scores = sigmoid_log_double_softmax(sim, z0, z1);

    return std::make_tuple(scores, sim);
}

torch::Tensor MatchAssignment::get_matchability(const torch::Tensor& desc) {
    return torch::sigmoid(matchability_->forward(desc)).squeeze(-1);
}

// Helper function to compute sigmoid log double softmax
// This would typically be in the main LightGlue class but is needed here
torch::Tensor MatchAssignment::sigmoid_log_double_softmax(
    const torch::Tensor& sim,
    const torch::Tensor& z0,
    const torch::Tensor& z1) {

    auto batch_size = sim.size(0);
    auto m = sim.size(1);
    auto n = sim.size(2);

    // Compute log sigmoid terms
    auto certainties = torch::log_sigmoid(z0) +
                       torch::log_sigmoid(z1).transpose(1, 2);

    // Compute log softmax scores
    auto scores0 = torch::log_softmax(sim, 2);
    auto scores1 = torch::log_softmax(
                       sim.transpose(-1, -2).contiguous(), 2)
                       .transpose(-1, -2);

    // Create output tensor
    auto scores = torch::full(
        {batch_size, m + 1, n + 1}, 0.0f,
        torch::TensorOptions().device(sim.device()).dtype(sim.dtype()));

    // Fill in the scores
    scores.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(torch::indexing::None, m),
         torch::indexing::Slice(torch::indexing::None, n)},
        scores0 + scores1 + certainties);

    // Handle unmatched points
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

    return scores;
}