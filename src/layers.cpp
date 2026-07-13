#include "layers.h"
#include "operators.h"
#include "utils.h"
#include <algorithm>

namespace simpledl {

Linear::Linear(int in_features, int out_features)
    : weight_({in_features, out_features}, true),
      bias_({out_features}, true)
{
    he_normal(weight_, in_features);
    std::fill(bias_.mutable_data(), bias_.mutable_data() + bias_.numel(), 0.0f);
}

Tensor Linear::forward(const Tensor& x) {
    Tensor out = matmul(x, weight_);
    return bias_add(out, bias_);
}

std::vector<Tensor> Linear::parameters() {
    return {weight_, bias_};
}

}  // namespace simpledl