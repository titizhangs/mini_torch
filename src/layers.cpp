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
    bias_.fill_data_ptr(0.0f);
}

Tensor Linear::forward(const Tensor& x) {
    Tensor out = matmul(x, weight_);
    return bias_add(out, bias_);
}

std::vector<Tensor> Linear::parameters() {
    return {weight_, bias_};
}

// ====== 新增 Conv2d 实现 ======
Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size,
               int stride, int padding, bool use_bias)
    : weight_({out_channels, in_channels, kernel_size, kernel_size}, true),
      bias_({out_channels}, true),
      stride_(stride),
      padding_(padding),
      use_bias_(use_bias)
{
    // He 初始化：fan_in = in_channels * kernel_h * kernel_w
    int fan_in = in_channels * kernel_size * kernel_size;
    he_normal(weight_, fan_in);
    bias_.fill_data_ptr(0.0f);
}

Tensor Conv2d::forward(const Tensor& x) {
    return conv2d(x, weight_, bias_, stride_, padding_);
}

std::vector<Tensor> Conv2d::parameters() {
    if (use_bias_) {
        return {weight_, bias_};
    }
    return {weight_};
}

// ====== 新增 MaxPool2d 实现 ======
MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {}

Tensor MaxPool2d::forward(const Tensor& x) {
    //未使用先注释
    // return max_pool2d(x, kernel_size_, stride_, padding_);
}

std::vector<Tensor> MaxPool2d::parameters() {
    return {}; // 池化层无可训练参数
}

}  // namespace simpledl