#include "tensor.h"
#include <algorithm>
#include <unordered_set>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <iomanip>

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

namespace {
// 分配指定设备的内存
void* alloc_memory(Device device, int64_t numel) {
    if (numel <= 0) return nullptr;
    size_t bytes = numel * sizeof(float);

    if (device == Device::kCPU) {
        return new float[numel](); // 初始化为0
    } else {
        void* ptr = nullptr;
        CHECK_CUDA(cudaMalloc(&ptr, bytes));
        CHECK_CUDA(cudaMemset(ptr, 0, bytes)); // 初始化为0
        return ptr;
    }
}

// 释放指定设备的内存
void free_memory(Device device, void* ptr) {
    if (!ptr) return;
    if (device == Device::kCPU) {
        delete[] static_cast<float*>(ptr);
    } else {
        CHECK_CUDA(cudaFree(ptr));
    }
}

// 跨设备数据拷贝
void copy_data(void* dst, Device dst_device,
               const void* src, Device src_device,
               int64_t numel) {
    if (numel <= 0 || !src || !dst) return;
    size_t bytes = numel * sizeof(float);

    cudaMemcpyKind kind;
    if (src_device == Device::kCPU && dst_device == Device::kCPU) {
        std::memcpy(dst, src, bytes);
        return;
    } else if (src_device == Device::kCPU && dst_device == Device::kCUDA) {
        kind = cudaMemcpyHostToDevice;
    } else if (src_device == Device::kCUDA && dst_device == Device::kCPU) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        kind = cudaMemcpyDeviceToDevice;
    }
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, kind));
}
// CUDA 核：逐元素填充指定值
__global__ void fill_kernel(float* dst, float value, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dst[idx] = value;
    }
}

// 通用填充函数：CPU/GPU统一接口
void fill_data(float* dst, Device device, float value, int64_t numel) {
    if (numel <= 0 || !dst) return;

    if (device == Device::kCPU) {
        std::fill_n(dst, numel, value);
    } else {
        const int block_size = 256;
        int64_t grid_size = (numel + block_size - 1) / block_size;
        fill_kernel<<<grid_size, block_size>>>(dst, value, numel);
        CHECK_CUDA(cudaGetLastError()); // 检查核函数启动错误
    }
}
} // anonymous namespace

// 内部实现结构体：仅本文件可见，外部完全无法访问
struct Tensor::Impl {
    std::vector<int64_t> shape_;
    // std::vector<float> data_;
    // std::vector<float> grad_;
    bool requires_grad_ = false;
    std::vector<std::shared_ptr<Impl>> inputs_;  // 强引用输入节点，保证生命周期
    BackwardFn backward_fn_;          // 反向传播函数
    size_t numel_; // 缓存元素总数，构造时一次性计算

    // ========== 新增：设备标记 ==========
    Device device_ = Device::kCPU;

    // ========== 改造：统一数据/梯度指针，替代原std::vector<float> ==========
    // 同一时间只在一个设备上有效，由 device_ 标记
    float* data_ptr_ = nullptr;   // 数据指针：CPU内存 或 GPU显存
    float* grad_ptr_ = nullptr;   // 梯度指针：懒分配，首次调用mutable_grad才分配

        // ========== 核心：Impl 析构函数，唯一负责释放内存 ==========
    ~Impl() {
        // 释放数据内存
        free_memory(device_, data_ptr_);
        free_memory(device_, grad_ptr_);
        
        data_ptr_ = nullptr;
        grad_ptr_ = nullptr;
    }

    Impl() = default;
    // ========== 关键：禁用 Impl 的拷贝，防止意外复制裸指针 ==========
    // 如果允许 Impl 直接拷贝，两个 Impl 会指向同一块内存，析构时重复释放
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // 移动构造/赋值可以保留，按需实现
    Impl(Impl&&) = default;
    Impl& operator=(Impl&&) = default;

    // 统一的元素数接口，内核层、反向执行都用它
    size_t numel() const {
        return numel_;
    }

    // 辅助：根据 shape 计算元素数
    static size_t calc_numel(const std::vector<int64_t>& shape) {
        // 计算元素总数，校验合法性
        size_t numel = 1;
        for (int64_t dim : shape) {
            if (dim <= 0) {
                throw std::invalid_argument("Tensor: dimension must be positive");
            }
            numel *= static_cast<size_t>(dim);
        }
        return numel;
    }
};

// -------------------- 构造函数 --------------------
Tensor::Tensor(const std::vector<int64_t>& shape, bool requires_grad, Device device)
    : impl_(std::make_shared<Impl>())
{
    impl_->shape_ = shape;
    impl_->requires_grad_ = requires_grad;
    impl_->device_ = device;
    impl_->numel_ = Impl::calc_numel(shape); // 构造时一次性计算

    // 分配数据内存
    impl_->data_ptr_ = static_cast<float*>(alloc_memory(device, impl_->numel_));
}

// // ========== 拷贝构造函数（深拷贝） ==========
// Tensor::Tensor(const Tensor& other)
//     : impl_(std::make_shared<Impl>(*other.impl_)) {
//     // 元数据已经被拷贝构造复制，单独拷贝数据
//     impl_->data_ptr_ = static_cast<float*>(alloc_memory(impl_->device_, impl_->numel_));
//     copy_data(impl_->data_ptr_, impl_->device_,
//               other.impl_->data_ptr_, other.impl_->device_,
//               impl_->numel_);
    
//     // 梯度有值才拷贝
//     if (other.impl_->grad_ptr_) {
//         impl_->grad_ptr_ = static_cast<float*>(alloc_memory(impl_->device_, impl_->numel_));
//         copy_data(impl_->grad_ptr_, impl_->device_,
//                   other.impl_->grad_ptr_, other.impl_->device_,
//                   impl_->numel_);
//     }
// }

Device Tensor::device() const {
    return impl_->device_;
}

Tensor Tensor::to(Device target) const {
    // 设备相同，直接返回深拷贝
    if (target == impl_->device_) {
        return Tensor(*this);
    }

    // 创建目标设备的新张量
    Tensor result(impl_->shape_,impl_->requires_grad_, target);
    // 拷贝数据
    copy_data(result.impl_->data_ptr_, target,
              impl_->data_ptr_, impl_->device_,
              impl_->numel_);
    
    // 有梯度则同步拷贝梯度
    if (impl_->grad_ptr_) {
        float* dst_grad = result.mutable_grad();
        copy_data(dst_grad, target,
                  impl_->grad_ptr_, impl_->device_,
                  impl_->numel_);
    }

    // 元数据同步
    result.impl_->requires_grad_ = impl_->requires_grad_;
    // 注意：计算图节点不跨设备拷贝，to之后反向传播需要重新构建
    return result;
}


// 真正操作Impl成员的逻辑，完全藏在cpp里
void Tensor::set_backward_impl(BackwardFn fn, std::vector<NodePtr> inputs) {
    impl_->backward_fn_ = std::move(fn);
    impl_->inputs_ = std::move(inputs);
}

// -------------------- 属性接口实现 --------------------
const std::vector<int64_t>& Tensor::shape() const {
    return impl_->shape_;
}

size_t Tensor::numel() const {
    return impl_->numel_;
}

bool Tensor::requires_grad() const {
    return impl_->requires_grad_;
}

// -------------------- 数据读写实现 --------------------
const float* Tensor::data() const {
    return impl_->data_ptr_;
}

float* Tensor::mutable_data()  const{
    return impl_->data_ptr_;
}

const float* Tensor::grad() const {
    return impl_->grad_ptr_;
}

float* Tensor::mutable_grad() const {
    return ensure_grad_allocated(impl_);
}



// -------------------- 反向传播核心 --------------------
void Tensor::backward() {
    // 第一步：DFS 后序遍历，得到拓扑有序节点列表
    std::vector<std::shared_ptr<Impl>> topo_order;
    std::unordered_set<std::shared_ptr<Impl>> visited;

    std::function<void(const std::shared_ptr<Impl>&)> dfs = [&](const std::shared_ptr<Impl>& node) {
        if (!node || visited.count(node)) return;
        visited.insert(node);
        for (const auto& input : node->inputs_) {
            dfs(input);
        }
        topo_order.push_back(node);
    };

    dfs(impl_);

    // 第二步：根节点（损失标量）梯度初始化为 1.0
    float* grad_ptr = mutable_grad();
    fill_data(grad_ptr, impl_->device_, 1.0f, impl_->numel_);

    // 第三步：逆拓扑序执行反向函数，梯度从输出向输入传播
    // 反向迭代器用 ++ 是标准正确写法：它内部重载了运算符，++ 等价于向容器头部移动，实现逆序遍历
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        NodePtr node = *it;
        if (node->backward_fn_) {
            node->backward_fn_(node->grad_ptr_,node->numel(),node->inputs_);

        }
        // std::vector<float> dst(node->numel_);
        // copy_data(dst.data(),Device::kCPU,node->grad_ptr_,node->device_,node->numel_);
        // std::cout<<&(*it) <<" 梯度：\n";
        // std::cout << std::fixed << std::setprecision(6);
        // for(int i=0;i<node->numel_;i++){
        //     std::cout<<dst[i]<<"\n";
        // }
    }
}
// 确保 Impl 的梯度内存已分配，返回可写指针
// 这是全项目唯一的梯度懒分配入口
float* Tensor::ensure_grad_allocated(NodePtr nodeptr) {
    if (!nodeptr) return nullptr;
    
    if (!nodeptr->requires_grad_) {
        throw std::runtime_error("Tensor: gradient is not enabled");
    }
    if (!nodeptr->grad_ptr_) {
        // 懒分配梯度内存
        nodeptr->grad_ptr_ = static_cast<float*>(alloc_memory(nodeptr->device_, nodeptr->numel_));
    }
    return nodeptr->grad_ptr_;
}

float* Tensor::mutable_grad(NodePtr nodeptr){
    return ensure_grad_allocated(nodeptr);
}
const float* Tensor::data(NodePtr nodeptr){
    return nodeptr->data_ptr_;
}


void Tensor::fill_grad_ptr(float value){
    fill_data(mutable_grad(), impl_->device_, value, impl_->numel_);
}

void Tensor::fill_data_ptr(float value){
    fill_data(impl_->data_ptr_, impl_->device_, value, impl_->numel_);
}

}  // namespace simpledl