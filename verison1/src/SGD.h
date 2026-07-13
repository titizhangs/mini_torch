#pragma once
class SGD {
public:
    float lr;
    std::vector<Tensor*> params;

    SGD(std::vector<Tensor*> parameters, float learning_rate)
        : params(std::move(parameters)), lr(learning_rate) {}

    // 一步更新：w = w - lr * grad
    void step() {
        for (Tensor* p : params) {
            std::vector<float>& data = p->data();
            const std::vector<float>& grad = p->grad();
            for (size_t i = 0; i < p->numel(); ++i) {
                data[i] -= lr * grad[i];
            }
        }
    }

    // 清零所有参数的梯度
    void zero_grad() {
        for (Tensor* p : params) {
            p->zero_grad();
        }
    }
};