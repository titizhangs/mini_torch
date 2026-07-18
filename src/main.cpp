#include <cstdio>
#include <vector>
#include "tensor.h"
#include "layers.h"
#include "operators.h"
#include "loss.h"
#include "optimizer.h"
#include "utils.h"
#include <unistd.h>
#include <cstdio>

using namespace simpledl;

int main() {
    char buf[1024];
    getcwd(buf, 1024);
    printf("Current work dir: %s\n", buf);
    // ===================== 1. 加载数据集 =====================
    // 只加载前2000张训练图、200张测试图，保证训练速度；可以自行调大
    const int train_num = 2000;
    const int test_num = 200;

    printf("Loading MNIST dataset...\n");
    Tensor train_images = load_mnist_images("train-images.idx3-ubyte", train_num);
    Tensor train_labels = load_mnist_labels("train-labels.idx1-ubyte", train_num);
    Tensor test_images  = load_mnist_images("t10k-images.idx3-ubyte", test_num);
    Tensor test_labels  = load_mnist_labels("t10k-labels.idx1-ubyte", test_num);

    printf("Train set: %lld images, shape: [%lld, %lld, %lld, %lld]\n",
           train_images.shape()[0],
           train_images.shape()[0], train_images.shape()[1],
           train_images.shape()[2], train_images.shape()[3]);

    // ===================== 构建现代标准CNN网络 =====================
    // 输入: [N, 1, 28, 28]
    // 现代设计：3x3核 + Same Padding + 步长卷积下采样
    // 通道数压低，保证CPU训练速度
    Conv2d conv1(1, 16, 3, 1, 1);   // [N,16,28,28]
    Conv2d conv2(16, 16, 3, 2, 1);  // 下采样 [N,16,14,14]
    Conv2d conv3(16, 32, 3, 1, 1);  // [N,32,14,14]
    Conv2d conv4(32, 32, 3, 2, 1);  // 下采样 [N,32,7,7]
    Linear fc(32 * 7 * 7, 10);

    // 收集所有可训练参数
    std::vector<Tensor> all_params;
    auto p1 = conv1.parameters();
    auto p2 = conv2.parameters();
    auto p3 = conv3.parameters();
    auto p4 = conv4.parameters();
    auto p_fc = fc.parameters();
    all_params.insert(all_params.end(), p1.begin(), p1.end());
    all_params.insert(all_params.end(), p2.begin(), p2.end());
    all_params.insert(all_params.end(), p3.begin(), p3.end());
    all_params.insert(all_params.end(), p4.begin(), p4.end());
    all_params.insert(all_params.end(), p_fc.begin(), p_fc.end());

    SGD optimizer(all_params, 0.01f);
    // ===================== 3. 训练 =====================
    const int epochs = 30;
    const int batch_size = 8;
    int total_batches = train_num / batch_size;

    printf("\nStart training...\n");
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float total_loss = 0.0f;

        //打印参数值
        // optimizer.show_params(); 

        for (int batch = 0; batch < total_batches; ++batch) {
            int start = batch * batch_size;
            int end = start + batch_size;

            // 切片取batch数据（简化版：直接构造batch张量）
            Tensor batch_img({batch_size, 1, 28, 28}, false);
            Tensor batch_label({batch_size, 10}, false);
            std::copy(train_images.data() + start * 28 * 28,
                      train_images.data() + end * 28 * 28,
                      batch_img.mutable_data());
            std::copy(train_labels.data() + start * 10,
                      train_labels.data() + end * 10,
                      batch_label.mutable_data());

            // 前向传播
            // 前向传播：全卷积下采样，无池化
            Tensor c1 = conv1.forward(batch_img);
            Tensor a1 = relu(c1);
            Tensor c2 = conv2.forward(a1);
            Tensor a2 = relu(c2);
            Tensor c3 = conv3.forward(a2);
            Tensor a3 = relu(c3);
            Tensor c4 = conv4.forward(a3);
            Tensor a4 = relu(c4);
            Tensor flat = flatten(a4);
            Tensor pred = fc.forward(flat);

            Tensor loss = mse_loss(pred, batch_label);
            total_loss += loss.data()[0];

            // 反向更新
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        float avg_loss = total_loss / total_batches;
        printf("Epoch %2d/%d | Avg Loss: %.6f\n", epoch, epochs, avg_loss);
    }

    // ===================== 4. 可视化测试：直观查看预测效果 =====================
    printf("\n===== Test set prediction demo =====\n");
    for (int i = 0; i < 15; ++i) { // 展示5张测试图
        printf("\n--- Test sample %d ---\n", i);
        print_image_ascii(test_images, i); // 打印字符画图像

        // 单样本前向预测
        Tensor single_img({1, 1, 28, 28}, false);
        std::copy(test_images.data() + i * 28 * 28,
                  test_images.data() + (i+1) * 28 * 28,
                  single_img.mutable_data());

        Tensor c1 = conv1.forward(single_img);
        Tensor a1 = relu(c1);
        Tensor c2 = conv2.forward(a1);
        Tensor a2 = relu(c2);
        Tensor c3 = conv3.forward(a2);
        Tensor a3 = relu(c3);
        Tensor c4 = conv4.forward(a3);
        Tensor a4 = relu(c4);
        Tensor flat = flatten(a4);
        Tensor pred = fc.forward(flat);

        int true_label = get_label_class(test_labels, i);
        int pred_label = get_label_class(pred, 0);
        printf("True label: %d | Predicted label: %d %s\n",
               true_label, pred_label,
               true_label == pred_label ? "[Correct]" : "[Wrong]");
    }

    return 0;
}