#include "utils.h"
#include <random>
#include <cmath>

namespace simpledl {

void he_normal(Tensor& tensor, int fan_in) {
    float std_dev = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, std_dev);

    float* data = tensor.mutable_data();
    size_t n = tensor.numel();
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

}  // namespace simpledl