#include "LightGlueModules.hpp"

TransformerLayer::TransformerLayer(int embed_dim, int num_heads, bool flash, bool bias) {
    // Initialize self-attention block
    self_attn_ = register_module("self_attn",
                                 std::make_shared<SelfBlock>(embed_dim, num_heads, flash, bias));

    // Initialize cross-attention block
    cross_attn_ = register_module("cross_attn",
                                  std::make_shared<CrossBlock>(embed_dim, num_heads, flash, bias));
}

std::tuple<torch::Tensor, torch::Tensor> TransformerLayer::forward(
    const torch::Tensor& desc0,
    const torch::Tensor& desc1,
    const torch::Tensor& encoding0,
    const torch::Tensor& encoding1,
    const torch::optional<torch::Tensor>& mask0,
    const torch::optional<torch::Tensor>& mask1) {

    if (mask0.has_value() && mask1.has_value())
    {
        return masked_forward(desc0, desc1, encoding0, encoding1,
                              mask0.value(), mask1.value());
    }

    // Apply self-attention independently to each descriptor set
    auto desc0_sa = self_attn_->forward(desc0, encoding0);
    auto desc1_sa = self_attn_->forward(desc1, encoding1);

    // Apply cross-attention between the two sets
    return cross_attn_->forward(desc0_sa, desc1_sa);
}

std::tuple<torch::Tensor, torch::Tensor> TransformerLayer::masked_forward(
    const torch::Tensor& desc0,
    const torch::Tensor& desc1,
    const torch::Tensor& encoding0,
    const torch::Tensor& encoding1,
    const torch::Tensor& mask0,
    const torch::Tensor& mask1) {

    // Create combined mask for cross attention
    auto mask = mask0 & mask1.transpose(-1, -2);

    // Create self-attention masks
    auto mask0_self = mask0 & mask0.transpose(-1, -2);
    auto mask1_self = mask1 & mask1.transpose(-1, -2);

    // Apply masked self-attention
    auto desc0_sa = self_attn_->forward(desc0, encoding0, mask0_self);
    auto desc1_sa = self_attn_->forward(desc1, encoding1, mask1_self);

    // Apply masked cross-attention
    return cross_attn_->forward(desc0_sa, desc1_sa, mask);
}