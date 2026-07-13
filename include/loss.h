#pragma once
#include "tensor.h"

namespace simpledl {

// 均方误差损失
Tensor mse_loss(const Tensor& pred, const Tensor& target);

}  // namespace simpledl