#include "LightGlueModules.hpp"
#include <torch/torch.h>

SelfBlock::SelfBlock(int embed_dim, int num_heads, bool flash, bool bias)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      Wqkv_(torch::nn::Linear(torch::nn::LinearOptions(embed_dim, 3 * embed_dim).bias(bias))),
      inner_attn_(std::make_shared<Attention>(flash)),
      out_proj_(torch::nn::Linear(torch::nn::LinearOptions(embed_dim, embed_dim).bias(bias))),
      ffn_(torch::nn::Sequential(
          torch::nn::Linear(torch::nn::LinearOptions(2 * embed_dim, 2 * embed_dim).bias(bias)),
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({2 * embed_dim}).elementwise_affine(true)),
          torch::nn::GELU(),
          torch::nn::Linear(torch::nn::LinearOptions(2 * embed_dim, embed_dim).bias(bias)))) {
    register_module("Wqkv", Wqkv_);
    register_module("out_proj", out_proj_);
    register_module("ffn", ffn_);
}

torch::Tensor SelfBlock::rotate_half(const torch::Tensor& x) {
    auto x_split = x.unflatten(-1, {-1, 2});
    auto x1 = x_split.select(-1, 0);
    auto x2 = x_split.select(-1, 1);
    return torch::stack({-x2, x1}, -1).flatten(-2);
}

torch::Tensor SelfBlock::apply_cached_rotary_emb(
    const torch::Tensor& freqs,
    const torch::Tensor& t) {

    return (t * freqs.select(0, 0)) +
           (rotate_half(t) * freqs.select(0, 1));
}

torch::Tensor SelfBlock::forward(
    const torch::Tensor& x,
    const torch::Tensor& encoding) {

    // Project to QKV
    auto qkv = Wqkv_->forward(x);
    qkv = qkv.unflatten(-1, {num_heads_, -1, 3}).transpose(1, 2);

    // Split into query, key, value
    auto q = qkv.select(-1, 0);
    auto k = qkv.select(-1, 1);
    auto v = qkv.select(-1, 2);

    // Apply rotary embeddings

    std::cout << "encoding: " << encoding.sizes() << std::endl;
    std::cout << "encoding dims: " << encoding.dim() << std::endl;
    std::cout << "q: " << q.sizes() << std::endl;
    std::cout << "q dims: " << q.dim() << std::endl;
    q = apply_cached_rotary_emb(encoding, q);
    k = apply_cached_rotary_emb(encoding, k);

    // Apply attention
    auto context = inner_attn_->forward(q, k, v);

    // Project output and apply residual connection
    auto message = out_proj_->forward(
        context.transpose(1, 2).flatten(/*start_dim=*/-2));

    // Combine with input using ffn
    return x + ffn_->forward(torch::cat({x, message}, -1));
}

CrossBlock::CrossBlock(int embed_dim, int num_heads, bool flash, bool bias)
    : heads_(num_heads),
      scale_(1.0f / sqrt(embed_dim / num_heads)) {

    auto dim_head = embed_dim / num_heads;
    auto inner_dim = dim_head * num_heads;

    // Initialize projections using LinearOptions
    to_qk_ = register_module(
        "to_qk", torch::nn::Linear(torch::nn::LinearOptions(embed_dim, inner_dim).bias(bias)));
    to_v_ = register_module(
        "to_v", torch::nn::Linear(torch::nn::LinearOptions(embed_dim, inner_dim).bias(bias)));
    to_out_ = register_module(
        "to_out", torch::nn::Linear(torch::nn::LinearOptions(inner_dim, embed_dim).bias(bias)));

    // Initialize feed-forward network
    auto ffn = torch::nn::Sequential(
        torch::nn::Linear(torch::nn::LinearOptions(2 * embed_dim, 2 * embed_dim).bias(true)),
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({2 * embed_dim}).elementwise_affine(true)),
        torch::nn::GELU(),
        torch::nn::Linear(torch::nn::LinearOptions(2 * embed_dim, embed_dim).bias(true)));

    ffn_ = register_module("ffn", ffn);

    // Initialize flash attention if requested
    if (flash && torch::cuda::is_available())
    {
        flash_ = register_module("flash", std::make_shared<Attention>(true));
    }
}

std::tuple<torch::Tensor, torch::Tensor> CrossBlock::forward(
    const torch::Tensor& x0,
    const torch::Tensor& x1,
    const torch::optional<torch::Tensor>& mask) {

    // Project inputs
    auto qk0 = to_qk_->forward(x0);
    auto qk1 = to_qk_->forward(x1);
    auto v0 = to_v_->forward(x0);
    auto v1 = to_v_->forward(x1);

    // Reshape for attention
    auto reshape_for_attention = [this](torch::Tensor t) {
        return t.unflatten(-1, {heads_, -1}).transpose(1, 2);
    };

    qk0 = reshape_for_attention(qk0);
    qk1 = reshape_for_attention(qk1);
    v0 = reshape_for_attention(v0);
    v1 = reshape_for_attention(v1);

    torch::Tensor m0, m1;

    if (flash_ && x0.device().is_cuda())
    {
        // Use flash attention
        m0 = flash_->forward(qk0, qk1, v1);
        m1 = flash_->forward(qk1, qk0, v0);
    } else
    {
        // Manual attention computation
        qk0 = qk0 * sqrt(scale_);
        qk1 = qk1 * sqrt(scale_);

        auto sim = torch::einsum("bhid,bhjd->bhij", {qk0, qk1});

        if (mask.has_value())
        {
            sim.masked_fill_(~mask.value(), -INFINITY);
        }

        auto attn01 = torch::softmax(sim, -1);
        auto attn10 = torch::softmax(sim.transpose(-2, -1).contiguous(), -1);

        m0 = torch::einsum("bhij,bhjd->bhid", {attn01, v1});
        m1 = torch::einsum("bhji,bhjd->bhid",
                           {attn10.transpose(-2, -1), v0});

        if (mask.has_value())
        {
            m0 = m0.nan_to_num();
            m1 = m1.nan_to_num();
        }
    }

    // Project back to original dimensions
    auto project_out = [this](torch::Tensor t) {
        return to_out_->forward(t.transpose(1, 2).flatten(/*start_dim=*/-2));
    };

    m0 = project_out(m0);
    m1 = project_out(m1);

    // Apply FFN with residual connections
    auto out0 = x0 + ffn_->forward(torch::cat({x0, m0}, -1));
    auto out1 = x1 + ffn_->forward(torch::cat({x1, m1}, -1));

    return std::make_tuple(out0, out1);
}