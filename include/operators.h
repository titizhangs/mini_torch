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

}  // namespace simpledl