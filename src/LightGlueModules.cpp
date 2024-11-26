#include "LightGlueModules.hpp"

LearnableFourierPosEnc::LearnableFourierPosEnc(
    int M, int dim, torch::optional<int> F_dim, float gamma)
    : gamma_(gamma) {

    int f_dim = F_dim.value_or(dim);
    // Initialize Wr with normal distribution
    Wr_ = register_module("Wr",
                          torch::nn::Linear(torch::nn::LinearOptions(M, f_dim / 2).bias(false)));

    // Initialize weights according to the paper
    auto std = gamma_ * gamma_;
    torch::nn::init::normal_(Wr_->weight, 0.0, std);
}

torch::Tensor LearnableFourierPosEnc::forward(const torch::Tensor& x) {
    // Project and compute trig functions
    auto projected = Wr_->forward(x);
    auto cosines = torch::cos(projected);
    auto sines = torch::sin(projected);

    // Stack and reshape
    auto emb = torch::stack({cosines, sines}, 0).unsqueeze(-3);
    return emb.repeat_interleave(2, -1);
}

TokenConfidence::TokenConfidence(int dim) {
    // Build sequential module for token confidence
    torch::nn::Sequential token;
    token->push_back(torch::nn::Linear(dim, 1));
    token->push_back(torch::nn::Sigmoid());

    token_ = register_module("token", token);
}

std::tuple<torch::Tensor, torch::Tensor> TokenConfidence::forward(
    const torch::Tensor& desc0,
    const torch::Tensor& desc1) {

    return std::make_tuple(
        token_->forward(desc0.detach()).squeeze(-1),
        token_->forward(desc1.detach()).squeeze(-1));
}

Attention::Attention(bool allow_flash) {
    // TODO: fix this
    // enable_flash_ = allow_flash && FLASH_AVAILABLE;
    has_sdp_ = torch::cuda::is_available() &&
               torch::cuda::is_available(); // &&
                                            // torch::().major >= 8;

    // if (enable_flash_) {
    //     torch::cuda::set_device(torch::cuda::current_device());
    // }
}

torch::Tensor Attention::forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::optional<torch::Tensor>& mask) {

    // Handle empty tensors
    if (q.size(-2) == 0 || k.size(-2) == 0)
    {
        return q.new_zeros({*q.sizes().begin(), q.size(-2), v.size(-1)});
    }

    // Use scaled dot-product attention if available
    if (enable_flash_ && q.device().is_cuda())
    {
        if (has_sdp_)
        {
            auto args_q = q.to(torch::kHalf).contiguous();
            auto args_k = k.to(torch::kHalf).contiguous();
            auto args_v = v.to(torch::kHalf).contiguous();

            auto result = torch::scaled_dot_product_attention(
                args_q, args_k, args_v,
                mask.has_value() ? mask.value() : torch::optional<torch::Tensor>());

            result = result.to(q.dtype());
            return mask.has_value() ? result.nan_to_num() : result;
        }
    }

    // Fall back to manual implementation
    auto scale = 1.0f / sqrt(q.size(-1));
    auto sim = torch::einsum("...id,...jd->...ij", {q, k}) * scale;

    if (mask.has_value())
    {
        sim.masked_fill_(~mask.value(), -INFINITY);
    }

    auto attn = torch::softmax(sim, -1);
    return torch::einsum("...ij,...jd->...id", {attn, v});
}