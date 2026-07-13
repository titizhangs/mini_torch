#include "tensor.h"
#include <algorithm>
#include <unordered_set>
#include <stdexcept>

namespace simpledl {

// 内部实现结构体：仅本文件可见，外部完全无法访问
struct Tensor::Impl {
    std::vector<int64_t> shape_;
    std::vector<float> data_;
    std::vector<float> grad_;
    bool requires_grad_ = false;
    std::vector<std::shared_ptr<Impl>> inputs_;  // 强引用输入节点，保证生命周期
    std::function<void()> backward_fn_;          // 反向传播函数
};

// -------------------- 构造函数 --------------------
Tensor::Tensor(const std::vector<int64_t>& shape, bool requires_grad)
    : impl_(std::make_shared<Impl>())
{
    impl_->shape_ = shape;
    impl_->requires_grad_ = requires_grad;

    // 计算元素总数，校验合法性
    size_t numel = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("Tensor: dimension must be positive");
        }
        numel *= static_cast<size_t>(dim);
    }

    impl_->data_.resize(numel, 0.0f);
    if (requires_grad) {
        impl_->grad_.resize(numel, 0.0f);
    }
}

// -------------------- 属性接口实现 --------------------
const std::vector<int64_t>& Tensor::shape() const {
    return impl_->shape_;
}

size_t Tensor::numel() const {
    return impl_->data_.size();
}

bool Tensor::requires_grad() const {
    return impl_->requires_grad_;
}

// -------------------- 数据读写实现 --------------------
const float* Tensor::data() const {
    return impl_->data_.data();
}

float* Tensor::mutable_data() {
    return impl_->data_.data();
}

const float* Tensor::grad() const {
    return impl_->grad_.data();
}

float* Tensor::mutable_grad() const {
    if (!impl_->requires_grad_) {
        throw std::runtime_error("Tensor: gradient is not enabled");
    }
    return impl_->grad_.data();
}

// -------------------- 计算图构建 --------------------
void Tensor::set_backward_fn(std::function<void()> backward_fn, std::vector<Tensor> inputs) {
    impl_->backward_fn_ = std::move(backward_fn);
    impl_->inputs_.reserve(inputs.size());
    for (const auto& input : inputs) {
        impl_->inputs_.push_back(input.impl_);  // 强引用，引用计数 +1
    }
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
    std::fill(impl_->grad_.begin(), impl_->grad_.end(), 1.0f);

    // 第三步：逆拓扑序执行反向函数，梯度从输出向输入传播
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        if ((*it)->backward_fn_) {
            (*it)->backward_fn_();
        }
    }
}

}  // namespace simpledl