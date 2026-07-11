#pragma once
#include <vector>
#include <iostream>
#include <functional>

class Tensor {
public:
    // ---------- 构造与析构 ----------
    // 二维构造
    Tensor(int r, int c, bool requires_grad = false);
    // 通用多维构造
    Tensor(std::vector<int> shape, bool requires_grad = false);
    ~Tensor() = default;

    // ---------- 自动求导接口 ----------
    // 触发反向传播
    void backward();
    // 梯度清零
    void zero_grad();
    void set_backward_fn(std::function<void(const Tensor& upstream)> fn) {
//         std::move 本身不移动数据
// 它只是一个类型转换：把左值强制转成右值引用，仅此而已。真正的移动操作，是 std::function 的移动赋值运算符做的。std::move 只是告诉编译器：“这个对象我不要了，你可以用移动版本的赋值，省点事”。
        backward_fn_ = std::move(fn);
    }

    // ---------- 兼容二维接口 ----------
    int rows() const { return shape_[0]; }
    int cols() const { return shape_[1]; }
    float& at(int i, int j);
    const float& at(int i, int j) const;

    // ---------- 通用张量接口 ----------
    int dim() const { return shape_.size(); }
    int numel() const { return static_cast<int>(data_.size()); }
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& stride() const { return stride_; }
    
    float& at(std::initializer_list<int> indices);
    const float& at(std::initializer_list<int> indices) const;
    
    Tensor reshape(const std::vector<int>& new_shape) const;

    // ---------- 属性读写 ----------
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad);

    const std::vector<float>& data() const { return data_; }
    std::vector<float>& data(){ return data_; }
    std::vector<float>& grad() const;


private:
    // ---------- 核心维度存储 ----------
    // int rows_;
    // int cols_;
    // ---------- 核心维度存储 ----------
    std::vector<int> shape_;
    std::vector<int> stride_;
    
    bool requires_grad_;
    std::vector<float> data_;
    mutable std::vector<float> grad_ ;
    std::function<void(const Tensor& upstream)> backward_fn_;

    // ---------- 内部工具函数 ----------
    // 梯度内存懒加载兜底
    void ensure_grad() const;
    // 根据shape计算行优先stride
    static std::vector<int> compute_stride(const std::vector<int>& shape);
    // 索引转偏移
    int offset(const std::vector<int>& indices) const;

};
// 算子声明（逐元素算子天然支持任意维度）
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a,const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a,const Tensor& b);
Tensor matmul(const Tensor& a,const Tensor& b);
Tensor relu(const Tensor& x);