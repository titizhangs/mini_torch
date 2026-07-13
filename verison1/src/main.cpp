#include "Tensor.h"
#include "Linear.h"
#include "SGD.h"
#include "Loss.h"
int main() {
    // 1. 准备训练数据：4个样本，输入x，标签y
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y_data = {5.0f, 8.0f, 11.0f, 14.0f}; // 对应3x+2

    Tensor x({4, 1}, false);
    Tensor target({4, 1}, false);
    // 填充数据
    for (int i = 0; i < 4; ++i) {
        x.data()[i * 1 + 0] = x_data[i];
        target.data()[i * 1 + 0] = y_data[i];
    }

    // 2. 定义网络：1维输入 -> 1维输出的单层Linear
    Linear layer(1, 1);
    // 收集所有参数
    auto all_params = layer.parameters();

    // 3. 定义优化器
    SGD optimizer(all_params, 0.01f);
    Loss* lossinstance=new Loss();
    // 4. 训练循环
    int epochs = 4000;
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        // 前向传播
        Tensor pred = layer.forward(x);
        Tensor loss = lossinstance->mse_loss(pred, target);

        // 清零梯度 + 反向传播
        optimizer.zero_grad();
        loss.backward();

        // 更新参数
        optimizer.step();

        // 打印损失
        if (epoch % 50 == 0) {
            printf("Epoch %d, Loss: %.4f, weight: %.4f, bias: %.4f\n",
                   epoch, loss.data()[0],
                   layer.weight.data()[0], layer.bias.data()[0]);
        }
    }

    return 0;
}

