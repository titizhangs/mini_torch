#pragma once
#include "tensor.h"

namespace simpledl {

// He 正态初始化，适配 ReLU 激活
void he_normal(Tensor& tensor, int fan_in);

}  // namespace simpledl