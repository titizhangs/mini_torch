#include "loss.h"
#include "operators.h"
#include <stdexcept>

namespace simpledl {

//均方误差损失（Mean Squared Error，简称 MSE）
Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::invalid_argument("MSELoss: shape mismatch");
    }

    Tensor diff = sub(pred, target);
    Tensor sq = mul(diff, diff);
    Tensor total = sum(sq);
    float n = static_cast<float>(pred.numel());

    return scale(total, 1.0f / n);
}

}  // namespace simpledl