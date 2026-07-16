#include "operators.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace simpledl {

Tensor sub(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("sub: shape mismatch");
    }

    Tensor out(a.shape(), a.requires_grad() || b.requires_grad());
    size_t n = a.numel();
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.mutable_data();

    for (size_t i = 0; i < n; ++i) {
        out_data[i] = a_data[i] - b_data[i];
    }

    if (out.requires_grad()) {
        out.set_backward_fn([a, b, out]() {
            size_t n = out.numel();
            const float* out_grad = out.grad();
            if (a.requires_grad()) {
                float* a_grad = a.mutable_grad();
                for (size_t i = 0; i < n; ++i) {
                    a_grad[i] += out_grad[i];
                }
            }
            if (b.requires_grad()) {
                float* b_grad = b.mutable_grad();
                for (size_t i = 0; i < n; ++i) {
                    b_grad[i] -= out_grad[i];
                }
            }
        }, {a, b});
    }
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("mul: shape mismatch");
    }

    Tensor out(a.shape(), a.requires_grad() || b.requires_grad());
    size_t n = a.numel();
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.mutable_data();

    for (size_t i = 0; i < n; ++i) {
        out_data[i] = a_data[i] * b_data[i];
    }

    if (out.requires_grad()) {
        out.set_backward_fn([a, b, out]() {
            size_t n = out.numel();
            const float* out_grad = out.grad();
            const float* a_data = a.data();
            const float* b_data = b.data();
            if (a.requires_grad()) {
                float* a_grad = a.mutable_grad();
                for (size_t i = 0; i < n; ++i) {
                    a_grad[i] += out_grad[i] * b_data[i];
                }
            }
            if (b.requires_grad()) {
                float* b_grad = b.mutable_grad();
                for (size_t i = 0; i < n; ++i) {
                    b_grad[i] += out_grad[i] * a_data[i];
                }
            }
        }, {a, b});
    }
    return out;
}

Tensor sum(const Tensor& x) {
    Tensor out({1}, x.requires_grad());
    float total = 0.0f;
    size_t n = x.numel();
    const float* x_data = x.data();

    for (size_t i = 0; i < n; ++i) {
        total += x_data[i];
    }
    out.mutable_data()[0] = total;

    if (out.requires_grad()) {
        out.set_backward_fn([x, out]() {
            float g = out.grad()[0];
            size_t n = x.numel();
            float* x_grad = x.mutable_grad();
            for (size_t i = 0; i < n; ++i) {
                x_grad[i] += g;
            }
        }, {x});
    }
    return out;
}

Tensor scale(const Tensor& x, float scale_val) {
    Tensor out(x.shape(), x.requires_grad());
    size_t n = x.numel();
    const float* x_data = x.data();
    float* out_data = out.mutable_data();

    for (size_t i = 0; i < n; ++i) {
        out_data[i] = x_data[i] * scale_val;
    }

    if (out.requires_grad()) {
        out.set_backward_fn([x, scale_val, out]() {
            size_t n = out.numel();
            const float* out_grad = out.grad();
            float* x_grad = x.mutable_grad();
            for (size_t i = 0; i < n; ++i) {
                x_grad[i] += out_grad[i] * scale_val;
            }
        }, {x});
    }
    return out;
}

Tensor matmul(const Tensor& x, const Tensor& weight) {
    if (x.shape().size() != 2 || weight.shape().size() != 2) {
        throw std::invalid_argument("matmul: only support 2D tensor");
    }

    int M = static_cast<int>(x.shape()[0]);
    int K = static_cast<int>(x.shape()[1]);
    int K2 = static_cast<int>(weight.shape()[0]);
    int N = static_cast<int>(weight.shape()[1]);
    if (K != K2) {
        throw std::invalid_argument("matmul: inner dimension mismatch");
    }

    Tensor out({M, N}, x.requires_grad() || weight.requires_grad());
    const float* x_data = x.data();
    const float* w_data = weight.data();
    float* out_data = out.mutable_data();
    std::fill(out_data, out_data + M * N, 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float xv = x_data[i * K + k];
            for (int j = 0; j < N; ++j) {
                out_data[i * N + j] += xv * w_data[k * N + j];
            }
        }
    }

    if (out.requires_grad()) {
        out.set_backward_fn([x, weight, out]() {
            const float* out_grad = out.grad();

            int M = static_cast<int>(x.shape()[0]);
            int K = static_cast<int>(x.shape()[1]);
            int N = static_cast<int>(weight.shape()[1]);

            if (x.requires_grad()) {
                float* x_grad = x.mutable_grad();
                // dX = dY @ W^T
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        float g = out_grad[i * N + j];
                        for (int k = 0; k < K; ++k) {
                            x_grad[i * K + k] += g * weight.data()[k * N + j];
                        }
                    }
                }
            }

            if (weight.requires_grad()) {
                float* w_grad = weight.mutable_grad();
                // dW = X^T @ dY
                for (int k = 0; k < K; ++k) {
                    for (int j = 0; j < N; ++j) {
                        float g = 0.0f;
                        for (int i = 0; i < M; ++i) {
                            g += x.data()[i * K + k] * out_grad[i * N + j];
                        }
                        w_grad[k * N + j] += g;
                    }
                }
            }
        }, {x, weight});
    }
    return out;
}

Tensor bias_add(const Tensor& x, const Tensor& bias) {
    if (x.shape().size() != 2 || bias.shape().size() != 1) {
        throw std::invalid_argument("bias_add: invalid shape");
    }

    int M = static_cast<int>(x.shape()[0]);
    int N = static_cast<int>(x.shape()[1]);
    if (bias.shape()[0] != N) {
        throw std::invalid_argument("bias_add: bias dimension mismatch");
    }

    Tensor out(x.shape(), x.requires_grad() || bias.requires_grad());
    const float* x_data = x.data();
    const float* b_data = bias.data();
    float* out_data = out.mutable_data();

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out_data[i * N + j] = x_data[i * N + j] + b_data[j];
        }
    }

    if (out.requires_grad()) {
        out.set_backward_fn([x, bias, out]() {
            const float* out_grad = out.grad();

            int M = static_cast<int>(x.shape()[0]);
            int N = static_cast<int>(x.shape()[1]);

            if(x.requires_grad()){
                float* x_grad = x.mutable_grad();
                // 输入梯度直接回传
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        x_grad[i * N + j] += out_grad[i * N + j];
                    }
                }
            }

            if(bias.requires_grad()){
                float* b_grad = bias.mutable_grad();
                // 偏置梯度按通道求和
                for (int j = 0; j < N; ++j) {
                    float sum_g = 0.0f;
                    for (int i = 0; i < M; ++i) {
                        sum_g += out_grad[i * N + j];
                    }
                    b_grad[j] += sum_g;
                }
            }
        }, {x, bias});
    }
    return out;
}

Tensor relu(const Tensor& x) {
    Tensor out(x.shape(), x.requires_grad());
    size_t n = x.numel();
    const float* x_data = x.data();
    float* out_data = out.mutable_data();

    for (size_t i = 0; i < n; ++i) {
        out_data[i] = std::max(0.0f, x_data[i]);
    }

    if (out.requires_grad()) {
        out.set_backward_fn([x, out]() {
            size_t n = out.numel();
            const float* out_grad = out.grad();
            const float* x_data = x.data();
            float* x_grad = x.mutable_grad();

            for (size_t i = 0; i < n; ++i) {
                if (x_data[i] > 0.0f) {
                    x_grad[i] += out_grad[i];
                }
            }
        }, {x});
    }
    return out;
}


Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride, int padding)
{
    // ========== 形状校验 ==========
    if (input.shape().size() != 4 || weight.shape().size() != 4 || bias.shape().size() != 1) {
        throw std::invalid_argument("conv2d: invalid tensor dimension");
    }

    int N  = static_cast<int>(input.shape()[0]);
    int C_in  = static_cast<int>(input.shape()[1]);
    int H_in  = static_cast<int>(input.shape()[2]);
    int W_in  = static_cast<int>(input.shape()[3]);

    int C_out = static_cast<int>(weight.shape()[0]);
    int K_c   = static_cast<int>(weight.shape()[1]);
    int K_h   = static_cast<int>(weight.shape()[2]);
    int K_w   = static_cast<int>(weight.shape()[3]);

    if (C_in != K_c || static_cast<int>(bias.shape()[0]) != C_out) {
        throw std::invalid_argument("conv2d: channel mismatch");
    }
    if (stride <= 0 || padding < 0) {
        throw std::invalid_argument("conv2d: invalid stride/padding");
    }

    // 计算输出尺寸
    int H_out = (H_in + 2 * padding - K_h) / stride + 1;
    int W_out = (W_in + 2 * padding - K_w) / stride + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("conv2d: output size is non-positive");
    }

    // ========== 前向计算 ==========
    bool need_grad = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    Tensor out({N, C_out, H_out, W_out}, need_grad);

    const float* in_data = input.data();
    const float* w_data  = weight.data();
    const float* b_data  = bias.data();
    float* out_data      = out.mutable_data();

    int in_hw  = H_in * W_in;
    int in_chw = C_in * in_hw;
    int w_hw   = K_h * K_w;
    int w_chw  = C_in * w_hw;
    int out_hw = H_out * W_out;
    int out_chw = C_out * out_hw;

    // 朴素卷积实现：逐样本、逐通道、逐位置做乘加
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = b_data[oc];
                    for (int ic = 0; ic < C_in; ++ic) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            for (int kw = 0; kw < K_w; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    int in_idx = n * in_chw + ic * in_hw + ih * W_in + iw;
                                    int w_idx  = oc * w_chw + ic * w_hw + kh * K_w + kw;
                                    sum += in_data[in_idx] * w_data[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = n * out_chw + oc * out_hw + oh * W_out + ow;
                    out_data[out_idx] = sum;
                }
            }
        }
    }

    // ========== 反向传播 ==========
    if (need_grad) {
        out.set_backward_fn([input, weight, bias, stride, padding, out]() {
            const float* out_grad = out.grad();
            int N  = static_cast<int>(input.shape()[0]);
            int C_in  = static_cast<int>(input.shape()[1]);
            int H_in  = static_cast<int>(input.shape()[2]);
            int W_in  = static_cast<int>(input.shape()[3]);
            int C_out = static_cast<int>(weight.shape()[0]);
            int K_h   = static_cast<int>(weight.shape()[2]);
            int K_w   = static_cast<int>(weight.shape()[3]);
            int H_out = static_cast<int>(out.shape()[2]);
            int W_out = static_cast<int>(out.shape()[3]);

            int in_hw  = H_in * W_in;
            int in_chw = C_in * in_hw;
            int w_hw   = K_h * K_w;
            int w_chw  = C_in * w_hw;
            int out_hw = H_out * W_out;
            int out_chw = C_out * out_hw;

            // 1. 偏置梯度：按通道求和
            if (bias.requires_grad()) {
                float* b_grad = bias.mutable_grad();
                for (int oc = 0; oc < C_out; ++oc) {
                    float sum_g = 0.0f;
                    for (int n = 0; n < N; ++n) {
                        for (int oh = 0; oh < H_out; ++oh) {
                            for (int ow = 0; ow < W_out; ++ow) {
                                int idx = n * out_chw + oc * out_hw + oh * W_out + ow;
                                sum_g += out_grad[idx];
                            }
                        }
                    }
                    b_grad[oc] += sum_g;
                }
            }

            // 2. 权重梯度：输入与输出梯度做互相关
            if (weight.requires_grad()) {
                const float* in_data = input.data();
                float* w_grad = weight.mutable_grad();
                for (int oc = 0; oc < C_out; ++oc) {
                    for (int ic = 0; ic < C_in; ++ic) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            for (int kw = 0; kw < K_w; ++kw) {
                                float sum_g = 0.0f;
                                for (int n = 0; n < N; ++n) {
                                    for (int oh = 0; oh < H_out; ++oh) {
                                        for (int ow = 0; ow < W_out; ++ow) {
                                            int ih = oh * stride - padding + kh;
                                            int iw = ow * stride - padding + kw;
                                            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                                int out_idx = n * out_chw + oc * out_hw + oh * W_out + ow;
                                                int in_idx  = n * in_chw + ic * in_hw + ih * W_in + iw;
                                                sum_g += out_grad[out_idx] * in_data[in_idx];
                                            }
                                        }
                                    }
                                }
                                int w_idx = oc * w_chw + ic * w_hw + kh * K_w + kw;
                                w_grad[w_idx] += sum_g;
                            }
                        }
                    }
                }
            }

            // 3. 输入梯度：输出梯度与翻转卷积核做转置卷积
            if (input.requires_grad()) {
                const float* w_data = weight.data();
                float* in_grad = input.mutable_grad();
                for (int n = 0; n < N; ++n) {
                    for (int oc = 0; oc < C_out; ++oc) {
                        for (int oh = 0; oh < H_out; ++oh) {
                            for (int ow = 0; ow < W_out; ++ow) {
                                float g = out_grad[n * out_chw + oc * out_hw + oh * W_out + ow];
                                for (int ic = 0; ic < C_in; ++ic) {
                                    for (int kh = 0; kh < K_h; ++kh) {
                                        for (int kw = 0; kw < K_w; ++kw) {
                                            int ih = oh * stride - padding + kh;
                                            int iw = ow * stride - padding + kw;
                                            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                                int w_idx  = oc * w_chw + ic * w_hw + kh * K_w + kw;
                                                int in_idx = n * in_chw + ic * in_hw + ih * W_in + iw;
                                                in_grad[in_idx] += g * w_data[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }, {input, weight, bias});
    }

    return out;
}
Tensor max_pool2d(const Tensor& input, int kernel_size, int stride, int padding) {
    if (input.shape().size() != 4) {
        throw std::invalid_argument("max_pool2d: input must be 4D tensor");
    }
    if (kernel_size <= 0 || padding < 0) {
        throw std::invalid_argument("max_pool2d: invalid kernel/padding");
    }
    // 步长默认等于核大小（标准下采样）
    if (stride <= 0) stride = kernel_size;

    int N = static_cast<int>(input.shape()[0]);
    int C = static_cast<int>(input.shape()[1]);
    int H_in = static_cast<int>(input.shape()[2]);
    int W_in = static_cast<int>(input.shape()[3]);

    int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    Tensor out({N, C, H_out, W_out}, input.requires_grad());
    const float* in_data = input.data();
    float* out_data = out.mutable_data();

    int in_hw = H_in * W_in;
    int in_chw = C * in_hw;
    int out_hw = H_out * W_out;
    int out_chw = C * out_hw;

    // 记录最大值位置，用于反向传播
    std::vector<int> max_idx_h(out.numel());
    std::vector<int> max_idx_w(out.numel());

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float max_val = -INFINITY;
                    int best_h = -1, best_w = -1;
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;
                            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                int idx = n * in_chw + c * in_hw + ih * W_in + iw;
                                if (in_data[idx] > max_val) {
                                    max_val = in_data[idx];
                                    best_h = ih;
                                    best_w = iw;
                                }
                            }
                        }
                    }
                    int out_idx = n * out_chw + c * out_hw + oh * W_out + ow;
                    out_data[out_idx] = max_val;
                    max_idx_h[out_idx] = best_h;
                    max_idx_w[out_idx] = best_w;
                }
            }
        }
    }

    if (input.requires_grad()) {
        out.set_backward_fn([input, max_idx_h, max_idx_w, out, stride, padding, kernel_size]() {
            const float* out_grad = out.grad();
            float* in_grad = input.mutable_grad();

            int N = static_cast<int>(input.shape()[0]);
            int C = static_cast<int>(input.shape()[1]);
            int H_in = static_cast<int>(input.shape()[2]);
            int W_in = static_cast<int>(input.shape()[3]);
            int H_out = static_cast<int>(out.shape()[2]);
            int W_out = static_cast<int>(out.shape()[3]);

            int in_hw = H_in * W_in;
            int in_chw = C * in_hw;
            int out_hw = H_out * W_out;
            int out_chw = C * out_hw;

            // 仅将梯度回传到最大值对应的位置
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    for (int oh = 0; oh < H_out; ++oh) {
                        for (int ow = 0; ow < W_out; ++ow) {
                            int out_idx = n * out_chw + c * out_hw + oh * W_out + ow;
                            int ih = max_idx_h[out_idx];
                            int iw = max_idx_w[out_idx];
                            if (ih >= 0 && iw >= 0) {
                                int in_idx = n * in_chw + c * in_hw + ih * W_in + iw;
                                in_grad[in_idx] += out_grad[out_idx];
                            }
                        }
                    }
                }
            }
        }, {input});
    }

    return out;
}
Tensor flatten(const Tensor& x) {
    if (x.shape().empty()) {
        throw std::invalid_argument("flatten: input tensor is empty");
    }

    int64_t batch = x.shape()[0];
    int64_t feat_dim = static_cast<int64_t>(x.numel()) / batch;

    Tensor out({batch, feat_dim}, x.requires_grad());
    // 内存连续，直接拷贝数据
    std::copy(x.data(), x.data() + x.numel(), out.mutable_data());

    if (x.requires_grad()) {
        out.set_backward_fn([x, out]() {
            // 梯度直接按内存连续拷贝回原形状
            std::copy(out.grad(), out.grad() + out.numel(), x.mutable_grad());
        }, {x});
    }

    return out;
}
}  // namespace simpledl