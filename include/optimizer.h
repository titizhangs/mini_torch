#pragma once
#include "tensor.h"
#include <vector>

namespace simpledl {

// 随机梯度下降优化器
class SGD {
public:
    SGD(std::vector<Tensor> params, float lr);

    void zero_grad();  // 清空所有参数梯度
    void step();       // 执行一步参数更新

private:
    std::vector<Tensor> params_;
    float lr_;
};

}  // namespace simpledl