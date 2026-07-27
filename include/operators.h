#pragma once
#include "tensor.h"

namespace simpledl {
// todo 逐元素加法：out = a + b
// 逐元素减法：out = a - b
Tensor sub(const Tensor& a, const Tensor& b);

// 逐元素乘法：out = a * b
Tensor mul(const Tensor& a, const Tensor& b);
// todo 逐元素除法：out = a / b

// 全量求和，输出标量 shape {1}
Tensor sum(const Tensor& x);

// 标量缩放：out = x * scale_val
Tensor scale(const Tensor& x, float scale_val);

// 矩阵乘法：x [M, K] @ weight [K, N] -> out [M, N]
Tensor matmul(const Tensor& x, const Tensor& weight);

// 广播偏置加法：[M, N] + [N]
Tensor bias_add(const Tensor& x, const Tensor& bias);

// ReLU 激活函数
Tensor relu(const Tensor& x);

// ====== 新增 CNN 相关算子 ======
// 二维卷积（NCHW格式，方形核/步长/填充）
// input: [N, in_channels, H, W]
// weight: [out_channels, in_channels, kernel_size, kernel_size]
// bias: [out_channels]
Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride = 1, int padding = 0);

// // 二维最大池化（NCHW格式，方形核/步长/填充）
// // stride 默认等于 kernel_size（标准下采样配置）
// Tensor max_pool2d(const Tensor& input, int kernel_size, int stride = 0, int padding = 0);

// 张量展平：将任意维度张量展为二维 [N, *] -> [N, D]
// 用于卷积层衔接全连接层
Tensor flatten(const Tensor& x);

}  // namespace simpledl