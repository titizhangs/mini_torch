#pragma once
#include "Tensor.h"
#include "He.h"
// Linear 全连接层
// 公式：output = x @ weight + bias
// 输入形状：[batch_size, in_features]
// 权重形状：[in_features, out_features]
// 偏置形状：[out_features]
// 输出形状：[batch_size, out_features]
class Linear {
public:
    Tensor weight;
    Tensor bias;
    //如果想要支持自由链式写法，唯一根治方案就是重构 Tensor 底层为shared_ptr的 Impl 句柄模式，让计算图内部强持有所有输入节点。
    Tensor out;

    Linear(int in_features, int out_features)
        : weight({in_features, out_features}, true),
          bias({out_features}, true),
          out({out_features}, true)
    {
        // 权重用He正态初始化，适配ReLU
        he_normal(weight, in_features);
        // 偏置初始化为0
        bias.data().resize(bias.numel(), 0.0f);
    }

    // 前向计算
    Tensor forward(const Tensor& x) {
        out = matmul(x, weight);
        return bias_add(out, bias);
    }

    // 返回所有可训练参数的指针，供优化器使用
    std::vector<Tensor*> parameters() {
        return {&weight, &bias};
    }
};