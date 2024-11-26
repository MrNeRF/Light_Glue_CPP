#pragma once

#include <torch/torch.h>

// Learnable Fourier Positional Encoding
class LearnableFourierPosEnc : public torch::nn::Module {
public:
    LearnableFourierPosEnc(int M, int dim, torch::optional<int> F_dim = torch::nullopt, float gamma = 1.0);

    // Forward function returns the position encoding
    torch::Tensor forward(const torch::Tensor& x);

private:
    float gamma_;
    torch::nn::Linear Wr_{nullptr};
};

// Token Confidence Module
class TokenConfidence : public torch::nn::Module {
public:
    explicit TokenConfidence(int dim);

    // Returns confidence scores for both descriptors
    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& desc0,
        const torch::Tensor& desc1);

private:
    torch::nn::Sequential token_{nullptr};
};

// Attention Module
class Attention : public torch::nn::Module {
public:
    explicit Attention(bool allow_flash);

    torch::Tensor forward(
        const torch::Tensor& q,
        const torch::Tensor& k,
        const torch::Tensor& v,
        const torch::optional<torch::Tensor>& mask = torch::nullopt);

private:
    bool enable_flash_;
    bool has_sdp_;
};

// Self-Attention Block
class SelfBlock : public torch::nn::Module {
public:
    SelfBlock(int embed_dim, int num_heads, bool flash = false, bool bias = true);

    torch::Tensor apply_cached_rotary_emb(
        const torch::Tensor& freqs, const torch::Tensor& t);

    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& encoding,
        const torch::optional<torch::Tensor>& mask = torch::nullopt);

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    torch::nn::Linear Wqkv_{nullptr};
    std::shared_ptr<Attention> inner_attn_;
    torch::nn::Linear out_proj_{nullptr};
    torch::nn::Sequential ffn_{nullptr};
    torch::Tensor rotate_half(const torch::Tensor& x);
};

// Cross-Attention Block
class CrossBlock : public torch::nn::Module {
public:
    CrossBlock(int embed_dim, int num_heads, bool flash = false, bool bias = true);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& x0,
        const torch::Tensor& x1,
        const torch::optional<torch::Tensor>& mask = torch::nullopt);

private:
    int heads_;
    float scale_;
    torch::nn::Linear to_qk_{nullptr};
    torch::nn::Linear to_v_{nullptr};
    torch::nn::Linear to_out_{nullptr};
    torch::nn::Sequential ffn_{nullptr};
    std::shared_ptr<Attention> flash_;
};

// Transformer Layer combining Self and Cross attention
class TransformerLayer : public torch::nn::Module {
public:
    TransformerLayer(int embed_dim, int num_heads, bool flash = false, bool bias = true);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& desc0,
        const torch::Tensor& desc1,
        const torch::Tensor& encoding0,
        const torch::Tensor& encoding1,
        const torch::optional<torch::Tensor>& mask0 = torch::nullopt,
        const torch::optional<torch::Tensor>& mask1 = torch::nullopt);

    std::tuple<torch::Tensor, torch::Tensor> masked_forward(
        const torch::Tensor& desc0,
        const torch::Tensor& desc1,
        const torch::Tensor& encoding0,
        const torch::Tensor& encoding1,
        const torch::Tensor& mask0,
        const torch::Tensor& mask1);

private:
    std::shared_ptr<SelfBlock> self_attn_;
    std::shared_ptr<CrossBlock> cross_attn_;
};

// Match Assignment Module
class MatchAssignment : public torch::nn::Module {
public:
    explicit MatchAssignment(int dim);

    torch::Tensor sigmoid_log_double_softmax(
        const torch::Tensor& sim,
        const torch::Tensor& z0,
        const torch::Tensor& z1);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& desc0,
        const torch::Tensor& desc1);

    torch::Tensor get_matchability(const torch::Tensor& desc);

private:
    int dim_;
    torch::nn::Linear matchability_{nullptr};
    torch::nn::Linear final_proj_{nullptr};
};