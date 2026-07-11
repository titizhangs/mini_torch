#include "Tensor.h"
#include <assert.h>
Tensor::Tensor(int r, int c, bool requires_grad)
: Tensor(std::vector<int>{r, c}, requires_grad){}

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
    // 开发期断言：不需要求导的张量禁止访问梯度，快速发现逻辑错误
    assert(requires_grad_ && "Error: Tensor does not require grad, cannot access gradient.");
    
    // 核心逻辑：没分配就分配，已分配就跳过
    if (grad_.empty()) {
        grad_.resize(rows() * cols(), 0.0f);
    }
}

// 可读写梯度接口
std::vector<float>& Tensor::grad() const {
    ensure_grad(); // 访问前强制兜底，100% 保证内存合法
    return grad_;
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

void Tensor::backward() {
    // 1. 懒加载梯度内存，确保 grad_ 已分配
    ensure_grad();

    // 2. 反向起点：只有标量张量可以直接调用 backward，梯度初始化为 1
    if (numel() != 1) {
        throw std::runtime_error("backward() can only be called on scalar (1-element) tensor");
    }
    grad_[0] = 1.0f;

    // 3. 如果绑定了反向函数，就调用它，把自己的梯度作为 upstream 传出去
    if (backward_fn_) {
        backward_fn_(*this);
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
    if (requires_grad_) {
        out.set_backward_fn([this](const Tensor& upstream) {
            // 上游梯度reshape回原形状，累加到当前梯度
            Tensor grad_in = upstream.reshape(shape_);
            // 累加梯度逻辑
            ensure_grad();
            for (int i = 0; i < numel(); ++i) {
                //反向传播的标准规则：一个张量可能被多个下游算子使用，梯度会从多条路径传回来，需要全部加在一起，所以用+
                grad_[i] += grad_in.data()[i];
            }
        });
    }
    return out;
}
Tensor add(const Tensor& a,const Tensor& b) {
    // 1. shape 校验
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("Add: tensor shape mismatch");
    }

    // 2. 创建输出 Tensor，梯度由输入共同决定
    bool need_grad = a.requires_grad() || b.requires_grad();
    Tensor out(a.rows(), a.cols(), need_grad);


    // 3. 前向计算
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            out.at(i, j) = a.at(i, j) + b.at(i, j);
        }
    }

    // 4. 仅在需要梯度时设置反向函数
    if (need_grad) {
        out.set_backward_fn([&a, &b](const Tensor& upstream) {
            int rows = upstream.rows();
            int cols = upstream.cols();

            // 合并循环，避免两次重复遍历
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    int idx = i * cols + j;
                    float grad_val = upstream.grad()[idx];

                    if (a.requires_grad()) {
                        a.grad()[idx] += grad_val;
                    }
                    if (b.requires_grad()) {
                        b.grad()[idx] += grad_val;
                    }
                }
            }
        });
    }

    return out;
}

Tensor sub(const Tensor& a,const Tensor& b) {
    // 1. shape 校验
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("Add: tensor shape mismatch");
    }

    // 2. 创建输出 Tensor，梯度由输入共同决定
    bool need_grad = a.requires_grad() || b.requires_grad();
    Tensor out(a.rows(), a.cols(), need_grad);


    // 3. 前向计算
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            out.at(i, j) = a.at(i, j) - b.at(i, j);
        }
    }

    // 4. 仅在需要梯度时设置反向函数
    if (need_grad) {
        out.set_backward_fn([&a, &b](const Tensor& upstream) {
            int rows = upstream.rows();
            int cols = upstream.cols();

            // 合并循环，避免两次重复遍历
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    int idx = i * cols + j;
                    float grad_val = upstream.grad()[idx];

                    if (a.requires_grad()) {
                        a.grad()[idx] += grad_val;
                    }
                    if (b.requires_grad()) {
                        b.grad()[idx] -= grad_val;
                    }
                }
            }
        });
    }

    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    // 1. shape校验，和加法完全一致
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("Div: tensors must have the same shape");
    }

    // 2. 创建输出张量，梯度由输入决定
    bool need_grad = a.requires_grad() || b.requires_grad();
    Tensor out(a.rows(), a.cols(), need_grad);

    // 3. 前向计算：逐元素相除
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            out.at(i,j) = a.at(i,j) * b.at(i,j);
        }
    }

    // 4. 反向传播
    if (need_grad) {
        out.set_backward_fn([&a, &b](const Tensor& upstream) {
            int rows = upstream.rows();
            int cols = upstream.cols();
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    int idx = i * cols + j;
                    float grad_val = upstream.grad()[idx];
                    float a_val = a.data()[idx];
                    float b_val = b.data()[idx];

                    // 导数推导：y = a*b
                    // dy/da = b，dy/db = a
                    if (a.requires_grad()) {
                        a.grad()[idx] += grad_val * b_val;
                    }
                    if (b.requires_grad()) {
                        b.grad()[idx] += grad_val * a_val;
                    }
                }
            }
        });
    }

    return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
    // 1. shape校验，和加法完全一致
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("Div: tensors must have the same shape");
    }

    // 2. 创建输出张量，梯度由输入决定
    bool need_grad = a.requires_grad() || b.requires_grad();
    Tensor out(a.rows(), a.cols(), need_grad);

    // 3. 前向计算：逐元素相除
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            out.at(i,j) = a.at(i,j) / b.at(i,j);
        }
    }

    // 4. 反向传播
    if (need_grad) {
        out.set_backward_fn([&a, &b](const Tensor& upstream) {
            int rows = upstream.rows();
            int cols = upstream.cols();
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    int idx = i * cols + j;
                    float grad_val = upstream.grad()[idx];
                    float a_val = a.data()[idx];
                    float b_val = b.data()[idx];

                    // 导数推导：y = a/b
                    // dy/da = 1/b，dy/db = -a/(b²)
                    if (a.requires_grad()) {
                        a.grad()[idx] += grad_val / b_val;
                    }
                    if (b.requires_grad()) {
                        b.grad()[idx] += grad_val * (-a_val / (b_val * b_val));
                    }
                }
            }
        });
    }

    return out;
}


Tensor matmul(const Tensor& a,const Tensor& b) {
    // 1. shape 校验
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Matrix size mismatch");
    }

    // 2. 创建输出 Tensor，梯度由输入共同决定
    bool need_grad = a.requires_grad() || b.requires_grad();
    Tensor out(a.rows(), a.cols(), need_grad);


    // 3. 前向计算
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < b.cols(); ++j) {
            for(int k =0; k< a.cols();++k)
            out.at(i, j) += a.at(i,k)*b.at(k,j);
        }
    }

    // 4. 仅在需要梯度时设置反向函数
    if (need_grad) {
        out.set_backward_fn([&a, &b](const Tensor& upstream) {
            if(a.requires_grad()){
                 //为什么循环可以调换顺序，首先三层连续的中间无代码的循环本质上就是i*k*j个枚举的运算，问题变成了为什么可以调换顺序
                //  答：因为下面都是取值后都是累加计算，直觉上不会有影响，具体是否有相应的数学详细的判定界限待研究
                for (int i = 0; i < a.rows(); ++i) {
                    for (int k = 0; k < a.cols(); ++k) {
                        for(int j=0;j<b.cols();++j){
                            float grad_val = upstream.grad()[i * upstream.cols() + j];
                            std::cout<<grad_val<<"\n";
                            std::cout<<b.data()[k*b.cols()+j]<<"\n";
                            a.grad()[i * a.cols() + k] += grad_val * b.data()[k*b.cols()+j];
                            std::cout<<i * a.cols() + k<< a.grad()[i * a.cols() + k]<<"\n";
                        }
                    }
                }
            }
            if(b.requires_grad()){
                for (int k = 0; k < b.rows(); ++k) {
                    for (int j = 0; j < b.cols(); ++j) {
                        for(int i=0;i<a.rows();++i){
                            float grad_val = upstream.grad()[i * upstream.cols() + j];
                            b.grad()[k * b.cols() + j] += grad_val * a.data()[i*a.cols()+k];
                        }
                    }
                }
            }
            
        });
    }

    return out;
}

