#pragma once
#include "Tensor.h"
// ReLU 激活层
// 无参数的层，封装激活函数

class ReLU {
public:
    Tensor forward(const Tensor& x) {
        return relu(x);
    }

    // 无可训练参数，返回空列表
    std::vector<Tensor*> parameters() {
        return {};
    }
};