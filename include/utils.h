#pragma once
#include "tensor.h"

namespace simpledl {

// He 正态初始化，适配 ReLU 激活
void he_normal(Tensor& tensor, int fan_in);

// ====== 新增：MNIST 数据集加载 ======
// 加载MNIST图像文件，返回形状 [num, 1, 28, 28] 的张量
Tensor load_mnist_images(const char*  file_path, int max_num = -1);

// 加载MNIST标签文件，返回形状 [num, 10] 的one-hot标签张量
Tensor load_mnist_labels(const char*  file_path, int max_num = -1);

// ====== 新增：图像可视化 ======
// 将单张28x28图像以ASCII字符画打印到控制台
void print_image_ascii(const Tensor& images, int index);

// 从one-hot张量中获取指定样本的类别编号
int get_label_class(const Tensor& labels, int index);

}  // namespace simpledl