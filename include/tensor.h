#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <cstdint>

namespace simpledl {

// 设备类型，后续算子分发、数据传输都以此为准
enum class Device {
    kCPU = 0,
    kCUDA = 1,
};

// 轻量级张量句柄：PIMPL 架构，外部仅见接口，内部实现完全隐藏
class Tensor {
private:
    struct Impl;          // 内部实现前置声明，外部不可见
    using BackwardFn = std::function<void(const float* out_grad, size_t out_numel, std::vector<std::shared_ptr<Impl>> out_inputs)>;
    std::shared_ptr<Impl> impl_;
public:
    // 获取当前张量所在设备
    Device device() const;
    // 切换设备，返回新的张量（原张量不变，符合值语义）
    // 若目标设备与当前一致，直接返回深拷贝
    Tensor to(Device target) const;

    // ========== 新增：公有节点指针类型 + 静态访问接口 ==========
    // 对外暴露"节点指针"类型名，但Impl本身依然是私有、不可见的
    using NodePtr = std::shared_ptr<Impl>;
    // 构造：指定形状 + 是否需要梯度
    explicit Tensor(const std::vector<int64_t>& shape, bool requires_grad = false, Device device=Device::kCPU);

    // 默认拷贝/移动语义：句柄轻量拷贝，共享底层实现
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    // ========== 属性查询（只读，const 保证不修改对象） ==========
    const std::vector<int64_t>& shape() const;
    size_t numel() const;
    bool requires_grad() const;
    
    // ========== 数据读写接口 ==========
    const float* data() const;       // 只读数据指针
    float* mutable_data() const;           // 可写数据指针
    const float* grad() const;       // 只读梯度指针
    float* mutable_grad() const;           // 可写梯度指针

    // ========== 计算图构建接口 ==========
    // 设置反向函数与依赖输入，自动强引用持有所有输入节点
    // 可变参数构建反向函数
    template <typename... Tensors>
    void set_backward_fn(BackwardFn fn, Tensors&&... inputs) {
        // 折叠表达式提取所有输入的impl指针，组装成vector
        std::vector<NodePtr> input_nodes;
        input_nodes.reserve(sizeof...(inputs));
        (input_nodes.emplace_back(get_impl(std::forward<Tensors>(inputs))), ...);
        
        // 核心操作委托给私有实现函数（实现在cpp里）
        set_backward_impl(std::move(fn), std::move(input_nodes));
    }


    // ========== 反向传播入口 ==========
    void backward();

    static float* mutable_grad(NodePtr nodeptr);
    static const float* data(NodePtr nodeptr);
    static NodePtr get_impl(const Tensor& tensor){
        return tensor.impl_;
    }

private:
    // 私有辅助函数：声明在头文件，实现在cpp
    void set_backward_impl(BackwardFn fn, std::vector<NodePtr> inputs);
    // 私有静态辅助函数，仅内部使用，对外完全不可见
    static float* ensure_grad_allocated(NodePtr impl);
};

}  // namespace simpledl