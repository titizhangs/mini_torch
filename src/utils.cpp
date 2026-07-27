#include "utils.h"
#include <random>
#include <cmath>
#include <fstream>
#include <cstring>

// CUDA 调用统一错误检查宏，必加，否则显存越界/启动失败会静默出错
#define CHECK_CUDA(call)                                 \
    do {                                                 \
        cudaError_t err = call;                          \
        if (err != cudaSuccess) {                        \
            throw std::runtime_error(                    \
                "CUDA Error: " + std::string(cudaGetErrorString(err)) \
            );                                           \
        }                                                \
    } while(0)


namespace simpledl {

void he_normal(Tensor& tensor, int fan_in) {
    float std_dev = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, std_dev);

    float* data = tensor.mutable_data();
    size_t n = tensor.numel();
    float tmp_data[n];
    for (size_t i = 0; i < n; ++i) {
        tmp_data[i] = dist(rng);
    }
    if(tensor.device()==Device::kCPU){
        std::memcpy(data, tmp_data, sizeof(float)*n);
    }else{
        CHECK_CUDA(cudaMemcpy(data, tmp_data, sizeof(float)*n, cudaMemcpyHostToDevice));
    }
}

// ====== 辅助：大端字节序转换（MNIST文件为大端存储） ======
static uint32_t read_big_endian(std::ifstream& file) {
    uint32_t val;
    file.read(reinterpret_cast<char*>(&val), sizeof(val));
    return ((val & 0xFF) << 24) |
           ((val & 0xFF00) << 8) |
           ((val & 0xFF0000) >> 8) |
           ((val & 0xFF000000) >> 24);
}

// ====== 加载MNIST图像 ======
Tensor load_mnist_images(const char* file_path, int max_num) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Cannot open image file: ") + file_path);
    }

    uint32_t magic = read_big_endian(file);
    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }

    uint32_t num_images = read_big_endian(file);
    uint32_t rows = read_big_endian(file);
    uint32_t cols = read_big_endian(file);

    if (max_num > 0 && static_cast<uint32_t>(max_num) < num_images) {
        num_images = static_cast<uint32_t>(max_num);
    }

    Tensor out({static_cast<int64_t>(num_images), 1, static_cast<int64_t>(rows), static_cast<int64_t>(cols)}, false);
    float* data = out.mutable_data();
    float tmp_data[out.numel()];
    for (uint32_t i = 0; i < num_images; ++i) {
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                // 归一化到 [0, 1]
                tmp_data[i * rows * cols + r * cols + c] = static_cast<float>(pixel) / 255.0f;
            }
        }
    }
    if(out.device()==Device::kCPU){
        std::memcpy(data, tmp_data, sizeof(float)*out.numel());
    }else{
        CHECK_CUDA(cudaMemcpy(data, tmp_data, sizeof(float)*out.numel(), cudaMemcpyHostToDevice));
    }

    return out;
}

// ====== 加载MNIST标签（转one-hot） ======
Tensor load_mnist_labels(const char* file_path, int max_num) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Cannot open label file: ") + file_path);
    }

    uint32_t magic = read_big_endian(file);
    if (magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    uint32_t num_labels = read_big_endian(file);
    if (max_num > 0 && static_cast<uint32_t>(max_num) < num_labels) {
        num_labels = static_cast<uint32_t>(max_num);
    }

    Tensor out({static_cast<int64_t>(num_labels), 10}, false);
    float* data = out.mutable_data();
    float tmp_data[out.numel()];
    std::fill(tmp_data, tmp_data + out.numel(), 0.0f);


    for (uint32_t i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        tmp_data[i * 10 + label] = 1.0f; // one-hot编码
    }
    if(out.device()==Device::kCPU){
        std::memcpy(data, tmp_data, sizeof(float)*out.numel());
    }else{
        CHECK_CUDA(cudaMemcpy(data, tmp_data, sizeof(float)*out.numel(), cudaMemcpyHostToDevice));
    }
    return out;
}

// ====== ASCII打印单张图像 ======
void print_image_ascii(const Tensor& images, int index) {
    if (images.shape().size() != 4 || images.shape()[1] != 1) {
        printf("Only support single-channel 4D image tensor\n");
        return;
    }

    int H = static_cast<int>(images.shape()[2]);
    int W = static_cast<int>(images.shape()[3]);
    const float* data = images.data() + index * H * W;


    const char* gray_chars = " .-+*#@"; // 灰度从浅到深
    if(images.device()==Device::kCPU){
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float val;
                val = data[h * W + w];
                int idx = static_cast<int>(val * 6.0f);
                if (idx > 6) idx = 6;
                if (idx < 0) idx = 0;
                printf("%c", gray_chars[idx]);
            }
            printf("\n");
        }
    }
    else{
        float tmp_data[H * W];
        CHECK_CUDA(cudaMemcpy(tmp_data, data, H * W *sizeof(float), cudaMemcpyDeviceToHost));
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float val;
                val = tmp_data[h * W + w];
                int idx = static_cast<int>(val * 6.0f);
                if (idx > 6) idx = 6;
                if (idx < 0) idx = 0;
                printf("%c", gray_chars[idx]);
            }
            printf("\n");
        }
    }
}

// ====== 获取one-hot对应的类别编号 ======
int get_label_class(const Tensor& labels, int index) {
    const float* data = labels.data() + index * 10;
    int max_idx = 0;
    float tmp_data[10];
    if(labels.device()==Device::kCPU) std::memcpy(tmp_data,data,10*sizeof(float));
    else{
        CHECK_CUDA(cudaMemcpy(tmp_data, data, 10*sizeof(float), cudaMemcpyDeviceToHost));
    }
    float max_val = tmp_data[0];
    for (int i = 1; i < 10; ++i) {
        if (tmp_data[i] > max_val) {
            max_val = tmp_data[i];
            max_idx = i;
        }
    }
    return max_idx;
}
}  // namespace simpledl