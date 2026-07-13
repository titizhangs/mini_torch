#include "optimizer.h"
#include <algorithm>

namespace simpledl {

SGD::SGD(std::vector<Tensor> params, float lr)
    : params_(std::move(params)), lr_(lr) {}

void SGD::zero_grad() {
    for (auto& param : params_) {
        std::fill(param.mutable_grad(), param.mutable_grad() + param.numel(), 0.0f);
    }
}

void SGD::step() {
    for (auto& param : params_) {
        float* data = param.mutable_data();
        const float* grad = param.grad();
        size_t n = param.numel();
        for (size_t i = 0; i < n; ++i) {
            data[i] -= lr_ * grad[i];
        }
    }
}

}  // namespace simpledl