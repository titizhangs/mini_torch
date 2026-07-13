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

}  // namespace simpledl