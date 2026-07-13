#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <cstdint>

namespace simpledl {

// 轻量级张量句柄：PIMPL 架构，外部仅见接口，内部实现完全隐藏
class Tensor {
private:
    struct Impl;          // 内部实现前置声明，外部不可见
    std::shared_ptr<Impl> impl_;

public:
    // 构造：指定形状 + 是否需要梯度
    explicit Tensor(const std::vector<int64_t>& shape, bool requires_grad = false);

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
    float* mutable_data();           // 可写数据指针
    const float* grad() const;       // 只读梯度指针
    float* mutable_grad() const;           // 可写梯度指针

    // ========== 计算图构建接口 ==========
    // 设置反向函数与依赖输入，自动强引用持有所有输入节点
    void set_backward_fn(std::function<void()> backward_fn, std::vector<Tensor> inputs);

    // ========== 反向传播入口 ==========
    void backward();
};

}  // namespace simpledl