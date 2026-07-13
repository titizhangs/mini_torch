#include "He.h"
#include <random>
#include <cmath>

// 使用全局随机数生成器，实际项目可封装到工具类
static std::mt19937& get_global_rng() {
    static std::mt19937 rng(std::random_device{}());
    return rng;
}

// He 正态分布初始化
// 输入：tensor 待初始化的张量；fan_in 输入神经元数量（即该层的输入维度）
void he_normal(Tensor& tensor, int fan_in) {
    if (fan_in <= 0) {
        throw std::invalid_argument("He init: fan_in must be positive");
    }

    float std = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, std);

    size_t total = tensor.numel();
    std::vector<float> data = tensor.data();
    for (size_t i = 0; i < total; ++i) {
        data[i] = dist(get_global_rng());
    }
}

// He 均匀分布初始化
void he_uniform(Tensor& tensor, int fan_in) {
    if (fan_in <= 0) {
        throw std::invalid_argument("He init: fan_in must be positive");
    }

    float bound = std::sqrt(6.0f / fan_in);
    std::uniform_real_distribution<float> dist(-bound, bound);

    size_t total = tensor.numel();
    std::vector<float> data = tensor.data();
    for (size_t i = 0; i < total; ++i) {
        data[i] = dist(get_global_rng());
    }
}