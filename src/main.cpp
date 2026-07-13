#include <iostream>
#include <vector>
#include "tensor.h"
#include "layers.h"
#include "loss.h"
#include "optimizer.h"

// 仅在 .cpp 文件中使用命名空间，头文件绝对禁止
using namespace simpledl;

int main() {
    // 构造数据集：y = 2*x + 1
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y_data = {3.0f, 5.0f, 7.0f, 9.0f};

    Tensor input({4, 1}, false);
    Tensor target({4, 1}, false);
    for (int i = 0; i < 4; ++i) {
        input.mutable_data()[i] = x_data[i];
        target.mutable_data()[i] = y_data[i];
    }

    // 构建网络、优化器
    Linear layer(1, 1);
    SGD optimizer(layer.parameters(), 0.01f);

    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 前向传播：支持任意链式调用，自动管理生命周期
        Tensor pred = layer.forward(input);
        Tensor loss = mse_loss(pred, target);

        // 反向传播 + 参数更新
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (epoch % 50 == 0||epoch+1 == epochs) {
            printf("Epoch %d, Loss: %.4f, weight: %.4f, bias: %.4f\n",
                   epoch, loss.data()[0],
                   layer.parameters()[0].data()[0], layer.parameters()[1].data()[0]);
        }
    }

    return 0;
}