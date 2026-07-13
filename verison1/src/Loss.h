#pragma once
#include "Tensor.h"
class Loss{
public:
    // 通用多维构造
    Loss():diff_({2,2},true),
    sq_({2,2},true),
    total_({2,2},true){}
    ~Loss() = default;
    Tensor mse_loss(const Tensor& pred, const Tensor& target) {
        if (pred.shape() != target.shape()) {
            throw std::invalid_argument("MSELoss: shape mismatch");
        }

        // 1. 计算差值
        diff_ = sub(pred, target);
        // 2. 逐元素平方
        sq_ = mul(diff_, diff_);
        // 3. 求和后除以元素个数，得到均值
        total_ = sum(sq_);
        float n = static_cast<float>(pred.numel());
        
        // 标量缩放：除以元素个数
        Tensor loss({1}, total_.requires_grad());
        loss.data()[0] = total_.data()[0] / n;

        // 反向：乘1/n
        if (total_.requires_grad()) {
            loss.set_backward_fn(
                [this, n](const Tensor& upstream) {
                    float g = upstream.grad()[0] / n;
                    total_.mutable_grad()[0] += g;
                },
                {&total_}
            );
        }
        return loss;
    }
private:
    Tensor diff_;
    Tensor sq_;
    Tensor total_;

};