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

}  // namespace simpledl