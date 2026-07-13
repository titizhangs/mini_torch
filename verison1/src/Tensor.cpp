#include "Tensor.h"
#include <assert.h>
#include <unordered_set>
// ========== 核心：多维构造函数实现 ==========
Tensor::Tensor(std::vector<int> shape, bool requires_grad)
    : shape_(std::move(shape))   // 移动语义转移vector，避免拷贝
    , requires_grad_(requires_grad)
{
    // 1. 形状合法性校验：提前拦截非法输入，避免后续索引越界
    for (int dim_size : shape_) {
        if (dim_size <= 0) {
            throw std::invalid_argument(
                "Tensor shape error: all dimensions must be positive integers"
            );
        }
    }

    // 2. 计算行优先（C风格）内存布局的步长数组
    stride_ = compute_stride(shape_);

    // 3. 计算总元素数，初始化数据内存（默认零初始化，避免脏值）
    int total_elements = 1;
    for (int s : shape_) {
        total_elements *= s;
    }
    data_.assign(total_elements, 0.0f);

    // 4. 梯度内存暂不分配：遵循懒加载设计，只有开启求导且首次使用时才通过ensure_grad分配
    //    backward_fn_ 默认构造为空函数对象，无需额外处理
}

float& Tensor::at(int i, int j){
    return data_[i*cols()+j];
}
const float& Tensor::at(int i, int j) const{
    return data_[i*cols()+j];
}

// ==============================================
// 核心兜底方法：梯度内存初始化唯一入口
// ==============================================
void Tensor::ensure_grad() const {
    // Debug 模式生效，Release 编译自动删除，零开销
    assert(requires_grad_ && "Cannot allocate gradient: tensor does not require grad");

    // 核心逻辑：没分配就分配，已分配就跳过
    if (grad_.empty()) {
        grad_.resize(numel(), 0.0f);
    }
}

// 可读写梯度接口
std::vector<float>& Tensor::mutable_grad() const{
    ensure_grad(); // 访问前强制兜底，100% 保证内存合法
    return grad_;
}
// const版本：只读用，不分配内存
const std::vector<float>& Tensor::grad() const {
    // 这里不调用ensure_grad，因为const函数不能修改成员
    // 如果梯度还没分配，返回空vector，只读不会崩溃
    return grad_;
}
float& Tensor::at(std::initializer_list<int> indices){
    // 把初始化列表转成vector，复用已有的offset计算逻辑
    return data_[offset(std::vector<int>(indices.begin(), indices.end()))];
}
const float& Tensor::at(std::initializer_list<int> indices) const{
    // 把初始化列表转成vector，复用已有的offset计算逻辑
    return data_[offset(std::vector<int>(indices.begin(), indices.end()))];
}
// void Tensor::backward() {
//     // 1. 兜底：如果构造函数没初始化梯度，这里补做基础初始化（双保险）
//     if (requires_grad_ && grad_.empty()) {
//         grad_.resize(rows_ * cols_, 1.0f);
//     }

//     // // 2. 起点初始化：根节点梯度置为全1（dL/dL = 1）
//     // // 注意：这里用 = 赋值，不是 +=，因为这是反向传播的起点
//     // std::fill(grad_.begin(), grad_.end(), 1.0f);

//     // 3. 执行反向函数，把自身梯度传递给上游输入节点
//     if (backward_fn_) {
//         backward_fn_(*this);
//     }

//     // 【可选，完整递归反向传播】如果上游节点也有反向函数，递归调用
//     // 注：你当前的单层加法不需要，后续做多算子串联时需要加上
//     // 比如a是输入节点，调用a.backward()继续往更上游传
// }

// 核心：触发反向传播（拓扑排序 + 逆序回传梯度）
void Tensor::backward() const {
    // 1. DFS后序遍历，收集计算图所有节点，得到拓扑序（输入在前，输出在后）
    std::vector<const Tensor*> topo;
    std::unordered_set<const Tensor*> visited;

    std::function<void(const Tensor*)> dfs = [&](const Tensor* node) {
        //避免梯度重复
        if (visited.count(node)) return;
        visited.insert(node);
        // 先递归所有输入节点
        for (const Tensor* input : node->inputs_) {
            dfs(input);
        }
        // 后序加入列表
        topo.push_back(node);
    };
    dfs(this);

    // 2. 反转得到逆拓扑序（输出在前，输入在后）
    std::reverse(topo.begin(), topo.end());

    // 3. 标量损失自动初始化梯度为1（dLoss/dLoss = 1）
    if (numel() == 1) {
        mutable_grad()[0] = 1.0f;
    }

    // 4. 按逆拓扑序逐个执行反向函数，梯度逐层回传
    for (const Tensor* node : topo) {
        if (node->backward_fn_) {
            node->backward_fn_(*node);
        }
    }
}

//stride计算
std::vector<int> Tensor::compute_stride(const std::vector<int>& shape) {
    std::vector<int> stride(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
}
//多维索引转偏移
int Tensor::offset(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("indices dimension mismatch");
    }
    int off = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        off += indices[i] * stride_[i];
    }
    return off;
}
//reshape 实现（零拷贝视图，仅改变形状）
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    // 计算总元素数，支持-1自动推导
    int total = numel();
    int infer_idx = -1;
    int product = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            infer_idx = static_cast<int>(i);
        } else {
            product *= new_shape[i];
        }
    }
    std::vector<int> shape = new_shape;
    if (infer_idx != -1) {
        shape[infer_idx] = total / product;
    }

    Tensor out(shape, requires_grad_);
    out.data_ = data_; // 共享数据（浅拷贝）
    // 如果需要自动求导，反向时要处理reshape的梯度形状还原
    //????我理解是原形态转变，直接是输入原变量，原梯度是否直接赋值为原值就可以了
    if (requires_grad_) {
        out.set_backward_fn([this](const Tensor& upstream) {
            for (size_t i = 0; i < numel(); ++i) {
                //反向传播的标准规则：一个张量可能被多个下游算子使用，梯度会从多条路径传回来，需要全部加在一起，所以用+
                grad_[i] += upstream.grad()[i];
            }
        },this->inputs_);
    }

    return out;
}
Tensor add(const Tensor& a,const Tensor& b) {
    // 1. shape 校验
    if (a.shape()!= b.shape()) {
        throw std::invalid_argument("add: tensor shape mismatch");
    }

    // 2. 创建输出 Tensor，梯度由输入共同决定
    bool need_grad = a.requires_grad() || b.requires_grad();
    std::vector<int> out_shape=a.shape();
    Tensor out(out_shape, need_grad);


    // 3. 前向计算
    size_t total = out.numel();
    for (size_t idx=0;idx<total;idx++){
        out.data()[idx] = a.data()[idx] + b.data()[idx];
    }


    // 4. 仅在需要梯度时设置反向函数
    if (need_grad) {
        out.set_backward_fn([&a, &b](const Tensor& upstream) {
            // 合并循环，避免两次重复遍历
            for (size_t idx=0;idx<upstream.numel();idx++) {
                float grad_val = upstream.grad()[idx];
                if (a.requires_grad()) {
                    a.mutable_grad()[idx] += grad_val;
                }
                if (b.requires_grad()) {
                    b.mutable_grad()[idx] += grad_val;
                }
            }
        },{&a,&b});
    }

    return out;
}
Tensor sub(const Tensor& a,const Tensor& b) {
    // 1. shape 校验
    if (a.shape()!= b.shape()) {
        throw std::invalid_argument("sub: tensor shape mismatch");
    }

    // 2. 创建输出 Tensor，梯度由输入共同决定
    bool need_grad = a.requires_grad() || b.requires_grad();
    std::vector<int> out_shape=a.shape();
    Tensor out(out_shape, need_grad);


    // 3. 前向计算
    size_t total = out.numel();
    for (size_t idx=0;idx<total;idx++){
        out.data()[idx] = a.data()[idx] - b.data()[idx];
    }


    // 4. 仅在需要梯度时设置反向函数
    if (need_grad) {
        out.set_backward_fn([&a, &b](const Tensor& upstream) {
            // 合并循环，避免两次重复遍历
            for (size_t idx=0;idx<upstream.numel();idx++) {
                float grad_val = upstream.grad()[idx];
                if (a.requires_grad()) {
                    a.mutable_grad()[idx] += grad_val;
                }
                if (b.requires_grad()) {
                    b.mutable_grad()[idx] -= grad_val;
                }
            }
        },{&a,&b});
    }

    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    // 1. shape 校验
    if (a.shape()!= b.shape()) {
        throw std::invalid_argument("mul: tensor shape mismatch");
    }

    // 2. 创建输出 Tensor，梯度由输入共同决定
    bool need_grad = a.requires_grad() || b.requires_grad();
    std::vector<int> out_shape=a.shape();
    Tensor out(out_shape, need_grad);


    // 3. 前向计算
    size_t total = out.numel();
    for (size_t idx=0;idx<total;idx++){
        out.data()[idx] = a.data()[idx] * b.data()[idx];
    }


    // 4. 仅在需要梯度时设置反向函数
    if (need_grad) {
        out.set_backward_fn([&a, &b](const Tensor& upstream) {
            // 合并循环，避免两次重复遍历
            for (size_t idx=0;idx<upstream.numel();idx++) {
                float grad_val = upstream.grad()[idx];
                // 导数推导：y = a*b
                // dy/da = b，dy/db = a
                if (a.requires_grad()) {
                    a.mutable_grad()[idx] += grad_val*b.data()[idx];
                }
                if (b.requires_grad()) {
                    b.mutable_grad()[idx] += grad_val*a.data()[idx];
                }
            }
        },{&a,&b});
    }

    return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
    // 1. shape 校验
    if (a.shape()!= b.shape()) {
        throw std::invalid_argument("div: tensor shape mismatch");
    }

    // 2. 创建输出 Tensor，梯度由输入共同决定
    bool need_grad = a.requires_grad() || b.requires_grad();
    std::vector<int> out_shape=a.shape();
    Tensor out(out_shape, need_grad);


    // 3. 前向计算
    size_t total = out.numel();
    // 如果 b 中存在 0 元素，前向会得到 inf，反向梯度也会爆炸。工业实现通常会加一个极小值 eps 兜底，或者显式抛出异常：
    float eps = 1e-8f;
    for (size_t idx=0;idx<total;idx++){
        out.data()[idx] = a.data()[idx] / (b.data()[idx]+eps);
    }


    // 4. 仅在需要梯度时设置反向函数
    if (need_grad) {
        out.set_backward_fn([&a, &b](const Tensor& upstream) {
            // 合并循环，避免两次重复遍历
            for (size_t idx=0;idx<upstream.numel();idx++) {
                float grad_val = upstream.grad()[idx];
                float a_val= a.data()[idx];
                float b_val= b.data()[idx];
                // 导数推导：y = a/b
                // dy/da = 1/b，dy/db = -a/(b²)
                if (a.requires_grad()) {
                    a.mutable_grad()[idx] += grad_val*(1/b_val);
                }
                if (b.requires_grad()) {
                    b.mutable_grad()[idx] += grad_val*(-a_val / (b_val*b_val));
                }
            }
        },{&a,&b});
    }

    return out;
}

//只是2维张量也就是矩阵的，张量推广待定
Tensor matmul(const Tensor& a, const Tensor& b) {
    auto shape_a = a.shape();
    auto shape_b = b.shape();
    if (shape_a.size() != 2 || shape_b.size() != 2) {
        throw std::invalid_argument("Matmul: only support 2D tensors");
    }

    // 维度常量：大写M/N/K，行业标准
    const int M = shape_a[0];  // A的行数、C的行数
    const int K = shape_a[1];  // A的列数、B的行数（收缩维度）
    const int N = shape_b[1];  // B的列数、C的列数

    if (K != shape_b[0]) {
        throw std::invalid_argument("Matmul: inner dimension mismatch");
    }

    bool need_grad = a.requires_grad() || b.requires_grad();
    Tensor out({M, N}, need_grad);

    // 前向：索引双写 ii/jj/kk，和Aik*Bkj数学记号一一对应
    for (int ii = 0; ii < M; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; ++kk) {
                // A[ii, kk] * B[kk, jj]，和数学公式完全对应
                sum += a.data()[ii * K + kk] * b.data()[kk * N + jj];
            }
            out.data()[ii * N + jj] = sum;
        }
    }

    // 反向传播
    if (need_grad) {
        out.set_backward_fn([&a, &b, M, K, N](const Tensor& upstream) {
            const std::vector<float>& d_out = upstream.grad();
            const std::vector<float>& A = a.data();
            const std::vector<float>& B = b.data();

            // dA = dOut @ B^T
            if (a.requires_grad()) {
                std::vector<float>& d_a = a.mutable_grad();
                for (int ii = 0; ii < M; ++ii) {
                    for (int kk = 0; kk < K; ++kk) {
                        float sum = 0.0f;
                        for (int jj = 0; jj < N; ++jj) {
                            sum += d_out[ii * N + jj] * B[kk * N + jj];
                        }
                        d_a[ii * K + kk] += sum;
                    }
                }
            }

            // dB = A^T @ dOut
            if (b.requires_grad()) {
                std::vector<float>& d_b = b.mutable_grad();
                for (int kk = 0; kk < K; ++kk) {
                    for (int jj = 0; jj < N; ++jj) {
                        float sum = 0.0f;
                        for (int ii = 0; ii < M; ++ii) {
                            sum += A[ii * K + kk] * d_out[ii * N + jj];
                        }
                        d_b[kk * N + jj] += sum;
                    }
                }
            }
        },{&a,&b});
    }

    return out;
}

Tensor relu(const Tensor& x) {
    // 1. 创建输出张量：逐元素算子，shape 与输入完全一致
    bool need_grad = x.requires_grad();
    Tensor out(x.shape(), need_grad);

    // 2. 前向计算：逐元素 max(0, x)
    size_t total = x.numel();
    for (size_t idx = 0; idx < total; idx++) {
        float val = x.data()[idx];
        out.data()[idx] = val > 0.0f ? val : 0.0f;
    }

    // 3. 设置反向传播
    if (need_grad) {
        out.set_backward_fn([&x](const Tensor& upstream) {
            size_t total = x.numel();
            for (size_t idx = 0; idx < total; idx++) {
                // 仅输入值大于 0 时，透传上游梯度
                if (x.data()[idx] > 0.0f) {
                    x.mutable_grad()[idx] += upstream.grad()[idx];
                }
                // x <= 0 时梯度为 0，无需操作（梯度初始化为 0）
            }
        },{&x});
    }

    return out;
}

//二维张量广播加一维偏置
Tensor bias_add(const Tensor& input, const Tensor& bias) {
    auto input_shape = input.shape();
    auto bias_shape = bias.shape();
    if (input_shape.size() != 2 || bias_shape.size() != 1) {
        throw std::invalid_argument("BiasAdd: input must be 2D, bias must be 1D");
    }
    if (input_shape[1] != bias_shape[0]) {
        throw std::invalid_argument("BiasAdd: feature dimension mismatch");
    }

    int M = input_shape[0];
    int N = input_shape[1];
    bool need_grad = input.requires_grad() || bias.requires_grad();
    Tensor out(input_shape, need_grad);

    // 前向：逐行加偏置
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out.data()[i * N + j] = input.data()[i * N + j] + bias.data()[j];
        }
    }

    // 反向
    if (need_grad) {
        out.set_backward_fn(
            [&input, &bias, M, N](const Tensor& upstream) {
                const std::vector<float>& d_out = upstream.grad();

                // 回传给输入：梯度直接透传
                if (input.requires_grad()) {
                    std::vector<float>& d_input = input.mutable_grad();
                    for (size_t idx = 0; idx < upstream.numel(); ++idx) {

                        d_input[idx] += d_out[idx];
                    }
                }

                // 回传给偏置：按列求和（所有行的梯度累加到对应偏置）
                if (bias.requires_grad()) {
                    std::vector<float>& d_bias = bias.mutable_grad();
                    for (int j = 0; j < N; ++j) {
                        float sum = 0.0f;
                        for (int i = 0; i < M; ++i) {
                            sum += d_out[i * N + j];
                        }
                        d_bias[j] += sum;
                    }
                }
            },
            {&input, &bias}
        );
    }
    return out;
}
//sum：全局求和，输出标量
Tensor sum(const Tensor& x) {
    bool need_grad = x.requires_grad();
    Tensor out({1}, need_grad);

    // 前向：所有元素求和
    float total = 0.0f;
    for (size_t idx = 0; idx < x.numel(); ++idx) {
        total += x.data()[idx];
    }
    out.data()[0] = total;

    // 反向：每个元素的梯度都等于上游标量梯度（sum对每个元素的导数都是1）
    if (need_grad) {
        out.set_backward_fn(
            [&x](const Tensor& upstream) {
                float g = upstream.grad()[0];
                std::vector<float>& d_x = x.mutable_grad();
                for (size_t idx = 0; idx < x.numel(); ++idx) {
                    d_x[idx] += g;
                }
            },
            {&x}
        );
    }
    return out;
}
Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::invalid_argument("MSELoss: shape mismatch");
    }

    // 1. 计算差值
    Tensor diff = sub(pred, target);
    // 2. 逐元素平方
    Tensor sq = mul(diff, diff);
    // 3. 求和后除以元素个数，得到均值
    Tensor total = sum(sq);
    float n = static_cast<float>(pred.numel());
    
    // 标量缩放：除以元素个数
    Tensor loss({1}, total.requires_grad());
    loss.data()[0] = total.data()[0] / n;

    // 反向：乘1/n
    if (total.requires_grad()) {
        loss.set_backward_fn(
            [total, n](const Tensor& upstream) {
                float g = upstream.grad()[0] / n;
                total.mutable_grad()[0] += g;
            },
            {&total}
        );
    }
    return loss;
}