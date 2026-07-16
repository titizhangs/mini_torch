#pragma once
#include "tensor.h"
#include <vector>

namespace simpledl {

// 全连接层
class Linear {
public:
    Linear(int in_features, int out_features);

    // 前向计算
    Tensor forward(const Tensor& x);

    // 返回所有可训练参数
    std::vector<Tensor> parameters();

private:
    Tensor weight_;  // 成员变量尾下划线，Google 风格标准
    Tensor bias_;
};


// ====== 新增 CNN 层 ======
// 二维卷积层（现代工业标准设计）
// 默认约定：
// - 推荐使用 3x3 卷积核
// - 保持尺寸请设置 padding = kernel_size / 2（Same Padding）
// - 下采样请设置 stride = 2，配合 same padding
// - 后续接 BatchNorm 时请关闭 bias
class Conv2d {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, bool use_bias = true);

    Tensor forward(const Tensor& x);
    std::vector<Tensor> parameters();

private:
    Tensor weight_;
    Tensor bias_;
    int stride_;
    int padding_;
    bool use_bias_;
};

// 二维最大池化层（无可训练参数）
class MaxPool2d {
public:
    MaxPool2d(int kernel_size, int stride = 0, int padding = 0);
    Tensor forward(const Tensor& x);
    std::vector<Tensor> parameters();

private:
    int kernel_size_;
    int stride_;
    int padding_;
};

}  // namespace simpledl