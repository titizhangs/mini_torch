#include "operators.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>

// ========== CUDA 工具与配置 ==========
#define CHECK_CUDA(cmd) \
    do { \
        cudaError_t err = (cmd); \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)



constexpr int CUDA_BLOCK_SIZE = 256; // 线程块大小，业界常用256/512




namespace simpledl {
    void sum_forward_cuda(const float* input, int64_t numel, float* output);
    void sum_backward_cuda(const float* out_grad, float* input_grad, int64_t numel);
    void scale_forward_cuda(const float* input, float scale_val, int64_t numel, float* output);
    void scale_backward_cuda(const float* out_grad, float* input_grad, int64_t numel, float scale_val);
    void matmul_forward_cuda(const float* x_data,const float* w_data,float* out_data,int64_t M,int64_t K,int64_t N);
    void matmul_backward_cuda(const float* out_grad,bool x_req_grad,bool weight_req_grad
    ,float* x_grad,float* weight_grad,const float* x_data,const float* weight_data,int64_t M,int64_t K,int64_t N);
    void bias_add_forward_cuda(float* out_data,const float* x_data,const float* b_data,int64_t M,int64_t N);
    void bias_add_backward_cuda_x(float* x_grad,const float* out_grad,int64_t M,int64_t N);
    void bias_add_backward_cuda_bias(float* bias_grad,const float* out_grad,int64_t M, int64_t N);
    void relu_forward_cuda(float* out_data,const float* x_data,int64_t numel);
    void relu_backward_cuda(float* x_grad,const float* out_grad,const float* x_data,int64_t out_numel);
 __global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int padding,
    int H_out, int W_out);
__global__ void conv2d_bias_backward_kernel(
    const float* __restrict__ out_grad,
    float* __restrict__ bias_grad,
    int N, int C_out, int H_out, int W_out);
    __global__ void conv2d_weight_backward_kernel(
    const float* __restrict__ out_grad,
    const float* __restrict__ input,
    float* __restrict__ weight_grad,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int padding,
    int H_out, int W_out);
__global__ void conv2d_input_backward_kernel(
    const float* __restrict__ out_grad,
    const float* __restrict__ weight,
    float* __restrict__ input_grad,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int padding,
    int H_out, int W_out);

    // 前向：元素级减法 a - b
__global__ void sub_forward_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ out,
                                   size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}


// dL/da = dL/dout * 1  
// dL/db = dL/dout * (-1) 
__global__ void sub_backward_kernel(float* __restrict__ a_grad,
                                    float* __restrict__ b_grad,
                                    const float* __restrict__ out_grad,
                                    bool a_req, bool b_req,
                                    size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float g = out_grad[idx];
        if (a_req) {
            a_grad[idx] += g;
        }
        if (b_req) {
            b_grad[idx] -= g;
        }
    }
}

// 前向：元素级乘法 a * b
__global__ void mul_forward_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ out,
                                   size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}


// dL/da = dL/dout * b 
// dL/db = dL/dout * a
__global__ void mul_backward_kernel(float* __restrict__ a_grad,
                                    float* __restrict__ b_grad,
                                    const float* __restrict__ a_data,
                                    const float* __restrict__ b_data,
                                    const float* __restrict__ out_grad,
                                    bool a_req, bool b_req,
                                    size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float g = out_grad[idx];
        if (a_req) {
            a_grad[idx] += g*b_data[idx];
        }
        if (b_req) {
            b_grad[idx] += g*a_data[idx];
        }
    }
}
Tensor sub_cpu(const Tensor& a, const Tensor& b) {
    Tensor out(a.shape(), a.requires_grad() || b.requires_grad(),a.device());
    size_t n = a.numel();
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.mutable_data();

    for (size_t i = 0; i < n; ++i) {
        out_data[i] = a_data[i] - b_data[i];
    }

    if (out.requires_grad()) {
        bool a_req_grad=a.requires_grad();
        bool b_req_grad=b.requires_grad();
        out.set_backward_fn([a_req_grad,b_req_grad](const float* out_grad, size_t out_numel, std::vector<Tensor::NodePtr> out_inputs) {
            Tensor::NodePtr a=out_inputs[0];
            Tensor::NodePtr b=out_inputs[1];
            if (a_req_grad) {
                float* a_grad = Tensor::mutable_grad(a);
                for (size_t i = 0; i < out_numel; ++i) {
                    a_grad[i] += out_grad[i];
                }
            }
            if (b_req_grad) {
                float* b_grad = Tensor::mutable_grad(b);
                for (size_t i = 0; i < out_numel; ++i) {
                    b_grad[i] -= out_grad[i];
                }
            }
        }, a, b);
    }
    return out;
}

Tensor sub_cuda(const Tensor& a, const Tensor& b) {
    const Device device = a.device();
    const bool out_req_grad = a.requires_grad() || b.requires_grad();
    Tensor out(a.shape(), out_req_grad, device);

    const size_t n = a.numel();
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.mutable_data();

    const int grid_size = static_cast<int>((n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    sub_forward_kernel<<<grid_size, CUDA_BLOCK_SIZE>>>(a_data, b_data, out_data, n);
    CHECK_CUDA(cudaGetLastError()); // 检查核启动错误

    // 反向传播注册
    if (out_req_grad) {
        const bool a_req_grad = a.requires_grad();
        const bool b_req_grad = b.requires_grad();

        out.set_backward_fn(
            [a_req_grad, b_req_grad](const float* out_grad, size_t out_numel,std::vector<Tensor::NodePtr> out_inputs) {
                Tensor::NodePtr a_node = out_inputs[0];
                Tensor::NodePtr b_node = out_inputs[1];

                float* a_grad = a_req_grad ? Tensor::mutable_grad(a_node) : nullptr;
                float* b_grad = b_req_grad ? Tensor::mutable_grad(b_node) : nullptr;

                const int grid_size = static_cast<int>((out_numel + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
                sub_backward_kernel<<<grid_size, CUDA_BLOCK_SIZE>>>(
                    a_grad, b_grad, out_grad, a_req_grad, b_req_grad, out_numel
                );
                CHECK_CUDA(cudaGetLastError());
            },
            a, b
        );
    }

    return out;
}
Tensor sub(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("sub: shape mismatch");
    }
    if (a.device()==Device::kCPU&&b.device()==Device::kCPU) {
        return sub_cpu(a,b);
    }
    if (a.device()==Device::kCUDA&&b.device()==Device::kCUDA) {
        return sub_cuda(a,b);
    }
    throw std::invalid_argument("sub: device mismatch");

}



Tensor mul_cpu(const Tensor& a, const Tensor& b) {
    Tensor out(a.shape(), a.requires_grad() || b.requires_grad(),a.device());
    size_t n = a.numel();
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.mutable_data();

    for (size_t i = 0; i < n; ++i) {
        out_data[i] = a_data[i] * b_data[i];
    }

    if (out.requires_grad()) {
        bool a_req_grad=a.requires_grad();
        bool b_req_grad=b.requires_grad();
        out.set_backward_fn([a_req_grad,b_req_grad,a_numel=a.numel(),b_numel=b.numel()](const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            Tensor::NodePtr a=out_inputs[0];
            Tensor::NodePtr b=out_inputs[1];
            const float* a_data = Tensor::data(a);
            const float* b_data = Tensor::data(b);
            if (a_req_grad) {
                float* a_grad = Tensor::mutable_grad(a);
                for (size_t i = 0; i < a_numel; ++i) {
                    a_grad[i] += out_grad[i] * b_data[i];
                }
            }
            if (b_req_grad) {
                float* b_grad = Tensor::mutable_grad(b);
                for (size_t i = 0; i < b_numel; ++i) {
                    b_grad[i] += out_grad[i] * a_data[i];
                }
            }
        }, a, b);
    }
    return out;
}


Tensor mul_cuda(const Tensor& a, const Tensor& b) {
    const Device device = a.device();
    const bool out_req_grad = a.requires_grad() || b.requires_grad();
    Tensor out(a.shape(), out_req_grad, device);

    const size_t n = a.numel();
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.mutable_data();

    const int grid_size = static_cast<int>((n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    mul_forward_kernel<<<grid_size, CUDA_BLOCK_SIZE>>>(a_data, b_data, out_data, n);
    CHECK_CUDA(cudaGetLastError()); // 检查核启动错误

    // 反向传播注册
    if (out_req_grad) {
        const bool a_req_grad = a.requires_grad();
        const bool b_req_grad = b.requires_grad();

        out.set_backward_fn(
            [a_req_grad, b_req_grad](const float* out_grad, size_t out_numel,std::vector<Tensor::NodePtr> out_inputs) {
                Tensor::NodePtr a_node = out_inputs[0];
                Tensor::NodePtr b_node = out_inputs[1];

                float* a_grad = a_req_grad ? Tensor::mutable_grad(a_node) : nullptr;
                float* b_grad = b_req_grad ? Tensor::mutable_grad(b_node) : nullptr;
                const float* a_data = Tensor::data(a_node);
                const float* b_data = Tensor::data(b_node);
                const int grid_size = static_cast<int>((out_numel + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
                mul_backward_kernel<<<grid_size, CUDA_BLOCK_SIZE>>>(
                    a_grad, b_grad, a_data, b_data, out_grad, a_req_grad, b_req_grad, out_numel
                );
                CHECK_CUDA(cudaGetLastError());
            },
            a, b
        );
    }

    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("mul: shape mismatch");
    }
    if (a.device()==Device::kCPU&&b.device()==Device::kCPU) {
        return mul_cpu(a,b);
    }
    if (a.device()==Device::kCUDA&&b.device()==Device::kCUDA) {
        return mul_cuda(a,b);
    }
    throw std::invalid_argument("mul: device mismatch");
}
Tensor sum_cpu(const Tensor& x) {
    Tensor out({1}, x.requires_grad(),x.device());
    float total = 0.0f;
    size_t n = x.numel();
    const float* x_data = x.data();

    for (size_t i = 0; i < n; ++i) {
        total += x_data[i];
    }
    out.mutable_data()[0] = total;

    if (out.requires_grad()) {
        out.set_backward_fn([x_numel=x.numel()](const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            float g = out_grad[0];
            const Tensor::NodePtr x=out_inputs[0];
            float* x_grad = Tensor::mutable_grad(x);
            for (size_t i = 0; i < x_numel; ++i) {
                x_grad[i] += g;
            }
        }, x);
    }
    return out;
}
Tensor sum_cuda(const Tensor& x) {
    Tensor out({1}, x.requires_grad(),x.device());
    size_t n = x.numel();
    const float* x_data = x.data();
    float* out_data = out.mutable_data();

    // CUDA 前向计算
    sum_forward_cuda(x_data, n, out_data);

    if (out.requires_grad()) {
        out.set_backward_fn([x_numel = x.numel(), x_device = x.device()](
            const float* out_grad, 
            size_t out_numel, 
            const std::vector<Tensor::NodePtr>& out_inputs
        ) {
            const Tensor::NodePtr x_node = out_inputs[0];
            float* x_grad = Tensor::mutable_grad(x_node);

            // CUDA 反向计算
            sum_backward_cuda(out_grad, x_grad, x_numel);
        }, x);
    }
    return out;
}

Tensor sum(const Tensor& x) {

    if (x.device()==Device::kCPU) {
        return sum_cpu(x);
    }
    if (x.device()==Device::kCUDA) {
        return sum_cuda(x);
    }
    throw std::invalid_argument("sum: device mismatch");
}
Tensor scale_cpu(const Tensor& x, float scale_val) {
    Tensor out(x.shape(), x.requires_grad(),x.device());
    size_t n = x.numel();
    const float* x_data = x.data();
    float* out_data = out.mutable_data();

    for (size_t i = 0; i < n; ++i) {
        out_data[i] = x_data[i] * scale_val;
    }

    if (out.requires_grad()) {
        out.set_backward_fn([scale_val](const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            Tensor::NodePtr x=out_inputs[0];
            float* x_grad = Tensor::mutable_grad(x);
            for (size_t i = 0; i < out_numel; ++i) {
                x_grad[i] += out_grad[i] * scale_val;
            }
        }, x);
    }
    return out;
}
Tensor scale_cuda(const Tensor& x, float scale_val) {
    Tensor out(x.shape(), x.requires_grad(),x.device());
    size_t n = x.numel();
    const float* x_data = x.data();
    float* out_data = out.mutable_data();

    // CUDA 前向计算
    scale_forward_cuda(x_data, scale_val, n, out_data);

    if (out.requires_grad()) {
        out.set_backward_fn([x_numel = x.numel(), x_device = x.device(),scale_val](
            const float* out_grad, 
            size_t out_numel, 
            const std::vector<Tensor::NodePtr>& out_inputs
        ) {
            const Tensor::NodePtr x_node = out_inputs[0];
            float* x_grad = Tensor::mutable_grad(x_node);

            // CUDA 反向计算
            scale_backward_cuda(out_grad, x_grad, x_numel,scale_val);
        }, x);
    }
    return out;
}

Tensor scale(const Tensor& x, float scale_val) {

    if (x.device()==Device::kCPU) {
        return scale_cpu(x,scale_val);
    }
    if (x.device()==Device::kCUDA) {
        return scale_cuda(x,scale_val);
    }
    throw std::invalid_argument("sum: device mismatch");
}
Tensor matmul_cuda(const Tensor& x, const Tensor& weight) {
    int M = static_cast<int>(x.shape()[0]);
    int K = static_cast<int>(x.shape()[1]);
    int K2 = static_cast<int>(weight.shape()[0]);
    int N = static_cast<int>(weight.shape()[1]);
    if (K != K2) {
        throw std::invalid_argument("matmul: inner dimension mismatch");
    }

    Tensor out({M, N}, x.requires_grad() || weight.requires_grad(),x.device());
    const float* x_data = x.data();
    const float* w_data = weight.data();
    float* out_data = out.mutable_data();

    // CUDA 前向计算
    matmul_forward_cuda(x_data, w_data, out_data,M,K,N);

    if (out.requires_grad()) {
        out.set_backward_fn([M, K, N, x_req_grad=x.requires_grad(),weight_req_grad=weight.requires_grad()]
        (const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            Tensor::NodePtr x=out_inputs[0];
            Tensor::NodePtr weight=out_inputs[1];
            // CUDA 反向计算
            matmul_backward_cuda(out_grad, x_req_grad, weight_req_grad,
                Tensor::mutable_grad(x), Tensor::mutable_grad(weight),Tensor::data(x),Tensor::data(weight), M, K, N);
        }, x, weight);
    }
    return out;
}
Tensor matmul_cpu(const Tensor& x, const Tensor& weight) {
    int M = static_cast<int>(x.shape()[0]);
    int K = static_cast<int>(x.shape()[1]);
    int K2 = static_cast<int>(weight.shape()[0]);
    int N = static_cast<int>(weight.shape()[1]);
    if (K != K2) {
        throw std::invalid_argument("matmul: inner dimension mismatch");
    }

    Tensor out({M, N}, x.requires_grad() || weight.requires_grad(),x.device());
    const float* x_data = x.data();
    const float* w_data = weight.data();
    float* out_data = out.mutable_data();

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float xv = x_data[i * K + k];
            for (int j = 0; j < N; ++j) {
                out_data[i * N + j] += xv * w_data[k * N + j];
            }
        }
    }

    if (out.requires_grad()) {
        out.set_backward_fn([M, K, N, x_req_grad=x.requires_grad(),weight_req_grad=weight.requires_grad()]
        (const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            Tensor::NodePtr x=out_inputs[0];
            Tensor::NodePtr weight=out_inputs[1];
            if (x_req_grad) {
                float* x_grad = Tensor::mutable_grad(x);
                const float* weight_data= Tensor::data(weight);
                // dX = dY @ W^T
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        float g = out_grad[i * N + j];
                        for (int k = 0; k < K; ++k) {
                            x_grad[i * K + k] += g * weight_data[k * N + j];
                        }
                    }
                }
            }

            if (weight_req_grad) {
                float* w_grad = Tensor::mutable_grad(weight);
                const float* x_data= Tensor::data(x);
                // dW = X^T @ dY
                for (int k = 0; k < K; ++k) {
                    for (int j = 0; j < N; ++j) {
                        float g = 0.0f;
                        for (int i = 0; i < M; ++i) {
                            g += x_data[i * K + k] * out_grad[i * N + j];
                        }
                        w_grad[k * N + j] += g;
                    }
                }
            }
        }, x, weight);
    }
    return out;
}
Tensor matmul(const Tensor& x, const Tensor& weight) {
    if (x.shape().size() != 2 || weight.shape().size() != 2) {
        throw std::invalid_argument("matmul: only support 2D tensor");
    }
    if(x.device()==Device::kCPU&&weight.device()==Device::kCPU){
        return matmul_cpu(x, weight);
    }
    if(x.device()==Device::kCUDA&&weight.device()==Device::kCUDA){
        return matmul_cuda(x, weight);
    }
    throw std::invalid_argument("matmul: invalid Device");
}
Tensor bias_add_cpu(const Tensor& x, const Tensor& bias) {
    int M = static_cast<int>(x.shape()[0]);
    int N = static_cast<int>(x.shape()[1]);
    Tensor out(x.shape(), x.requires_grad() || bias.requires_grad(),x.device());
    const float* x_data = x.data();
    const float* b_data = bias.data();
    float* out_data = out.mutable_data();

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out_data[i * N + j] = x_data[i * N + j] + b_data[j];
        }
    }

    if (out.requires_grad()) {
        out.set_backward_fn([x_req_grad=x.requires_grad(),bias_req_grad=bias.requires_grad(),M,N]
        (const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            if(x_req_grad){
                Tensor::NodePtr x=out_inputs[0];
                float* x_grad = Tensor::mutable_grad(x);
                // 输入梯度直接回传
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        x_grad[i * N + j] += out_grad[i * N + j];
                    }
                }
            }

            if(bias_req_grad){
                Tensor::NodePtr bias=out_inputs[1];
                float* bias_grad = Tensor::mutable_grad(bias);
                // 偏置梯度按通道求和
                for (int j = 0; j < N; ++j) {
                    float sum_g = 0.0f;
                    for (int i = 0; i < M; ++i) {
                        sum_g += out_grad[i * N + j];
                    }
                    bias_grad[j] += sum_g;
                }
            }
        }, x, bias);
    }
    return out;
}
Tensor bias_add_cuda(const Tensor& x, const Tensor& bias) {
    int M = static_cast<int>(x.shape()[0]);
    int N = static_cast<int>(x.shape()[1]);
    Tensor out(x.shape(), x.requires_grad() || bias.requires_grad(),x.device());
    const float* x_data = x.data();
    const float* b_data = bias.data();
    float* out_data = out.mutable_data();

    bias_add_forward_cuda(out_data,x_data,b_data,M,N);

    if (out.requires_grad()) {
        out.set_backward_fn([x_req_grad=x.requires_grad(),bias_req_grad=bias.requires_grad(),M,N]
        (const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            if(x_req_grad){
                Tensor::NodePtr x=out_inputs[0];
                float* x_grad = Tensor::mutable_grad(x);
                bias_add_backward_cuda_x(x_grad,out_grad,M,N);
            }

            if(bias_req_grad){
                Tensor::NodePtr bias=out_inputs[1];
                float* bias_grad = Tensor::mutable_grad(bias);
                bias_add_backward_cuda_bias(bias_grad,out_grad,M,N);
            }
        }, x, bias);
    }
    return out;
}
Tensor bias_add(const Tensor& x, const Tensor& bias) {
    if (x.shape().size() != 2 || bias.shape().size() != 1) {
        throw std::invalid_argument("bias_add: invalid shape");
    }
    if (bias.shape()[0] != static_cast<int>(x.shape()[1])) {
        throw std::invalid_argument("bias_add: bias dimension mismatch");
    }
    if(x.device()==Device::kCPU&&bias.device()==Device::kCPU){
        return bias_add_cpu(x, bias);
    }
    if(x.device()==Device::kCUDA&&bias.device()==Device::kCUDA){
        return bias_add_cuda(x, bias);
    }
    throw std::invalid_argument("bias_add: invalid Device");
}
Tensor relu_cpu(const Tensor& x) {
    Tensor out(x.shape(), x.requires_grad(),x.device());
    size_t n = x.numel();
    const float* x_data = x.data();
    float* out_data = out.mutable_data();

    for (size_t i = 0; i < n; ++i) {
        out_data[i] = std::max(0.0f, x_data[i]);
    }

    if (out.requires_grad()) {
        out.set_backward_fn([](const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            Tensor::NodePtr x=out_inputs[0];
            const float* x_data = Tensor::data(x);
            float* x_grad = Tensor::mutable_grad(x);

            for (size_t i = 0; i < out_numel; ++i) {
                if (x_data[i] > 0.0f) {
                    x_grad[i] += out_grad[i];
                }
            }
        }, x);
    }
    return out;
}
Tensor relu_cuda(const Tensor& x) {
    Tensor out(x.shape(), x.requires_grad(),x.device());
    size_t n = x.numel();
    const float* x_data = x.data();
    float* out_data = out.mutable_data();

    relu_forward_cuda(out_data,x_data,n);

    if (out.requires_grad()) {
        out.set_backward_fn([](const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            Tensor::NodePtr x=out_inputs[0];
            const float* x_data = Tensor::data(x);
            float* x_grad = Tensor::mutable_grad(x);
            relu_backward_cuda(x_grad,out_grad,x_data,out_numel);
        }, x);
    }
    return out;
}
Tensor relu(const Tensor& x) {
    if(x.device()==Device::kCPU){
        return relu_cpu(x);
    }
    if(x.device()==Device::kCUDA){
        return relu_cuda(x);
    }
    throw std::invalid_argument("relu: invalid Device");
}

Tensor conv2d_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride, int padding)
{
    int N  = static_cast<int>(input.shape()[0]);
    int C_in  = static_cast<int>(input.shape()[1]);
    int H_in  = static_cast<int>(input.shape()[2]);
    int W_in  = static_cast<int>(input.shape()[3]);

    int C_out = static_cast<int>(weight.shape()[0]);
    int K_c   = static_cast<int>(weight.shape()[1]);
    int K_h   = static_cast<int>(weight.shape()[2]);
    int K_w   = static_cast<int>(weight.shape()[3]);

    if (C_in != K_c || static_cast<int>(bias.shape()[0]) != C_out) {
        throw std::invalid_argument("conv2d: channel mismatch");
    }
    if (stride <= 0 || padding < 0) {
        throw std::invalid_argument("conv2d: invalid stride/padding");
    }

    // 计算输出尺寸
    int H_out = (H_in + 2 * padding - K_h) / stride + 1;
    int W_out = (W_in + 2 * padding - K_w) / stride + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("conv2d: output size is non-positive");
    }

    // ========== 前向计算 ==========
    bool need_grad = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    Tensor out({N, C_out, H_out, W_out}, need_grad,input.device());

    const float* in_data = input.data();
    const float* w_data  = weight.data();
    const float* b_data  = bias.data();
    float* out_data      = out.mutable_data();

    int in_hw  = H_in * W_in;
    int in_chw = C_in * in_hw;
    int w_hw   = K_h * K_w;
    int w_chw  = C_in * w_hw;
    int out_hw = H_out * W_out;
    int out_chw = C_out * out_hw;

    // 朴素卷积实现：逐样本、逐通道、逐位置做乘加
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = b_data[oc];
                    for (int ic = 0; ic < C_in; ++ic) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            for (int kw = 0; kw < K_w; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    int in_idx = n * in_chw + ic * in_hw + ih * W_in + iw;
                                    int w_idx  = oc * w_chw + ic * w_hw + kh * K_w + kw;
                                    sum += in_data[in_idx] * w_data[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = n * out_chw + oc * out_hw + oh * W_out + ow;
                    out_data[out_idx] = sum;
                }
            }
        }
    }

    // ========== 反向传播 ==========
    if (need_grad) {
        out.set_backward_fn([bias_req_grad=bias.requires_grad(),weight_req_grad=weight.requires_grad(),input_req_grad=input.requires_grad(),
            stride, padding,N,C_in,H_in,W_in,C_out,K_h,K_w,H_out,W_out,in_hw,in_chw,w_hw,w_chw,out_hw,out_chw]
            (const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            Tensor::NodePtr input=out_inputs[0];
            Tensor::NodePtr weight=out_inputs[1];
            Tensor::NodePtr bias=out_inputs[2];
            // 1. 偏置梯度：按通道求和
            if (bias_req_grad) {

                float* b_grad = Tensor::mutable_grad(bias);
                for (int oc = 0; oc < C_out; ++oc) {
                    float sum_g = 0.0f;
                    for (int n = 0; n < N; ++n) {
                        for (int oh = 0; oh < H_out; ++oh) {
                            for (int ow = 0; ow < W_out; ++ow) {
                                int idx = n * out_chw + oc * out_hw + oh * W_out + ow;
                                sum_g += out_grad[idx];
                            }
                        }
                    }
                    b_grad[oc] += sum_g;
                }
            }

            // 2. 权重梯度：输入与输出梯度做互相关
            if (weight_req_grad) {
                const float* in_data = Tensor::data(input);
                float* w_grad = Tensor::mutable_grad(weight);
                for (int oc = 0; oc < C_out; ++oc) {
                    for (int ic = 0; ic < C_in; ++ic) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            for (int kw = 0; kw < K_w; ++kw) {
                                float sum_g = 0.0f;
                                for (int n = 0; n < N; ++n) {
                                    for (int oh = 0; oh < H_out; ++oh) {
                                        for (int ow = 0; ow < W_out; ++ow) {
                                            int ih = oh * stride - padding + kh;
                                            int iw = ow * stride - padding + kw;
                                            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                                int out_idx = n * out_chw + oc * out_hw + oh * W_out + ow;
                                                int in_idx  = n * in_chw + ic * in_hw + ih * W_in + iw;
                                                sum_g += out_grad[out_idx] * in_data[in_idx];
                                            }
                                        }
                                    }
                                }
                                int w_idx = oc * w_chw + ic * w_hw + kh * K_w + kw;
                                w_grad[w_idx] += sum_g;
                            }
                        }
                    }
                }
            }

            // 3. 输入梯度：输出梯度与翻转卷积核做转置卷积
            if (input_req_grad) {
                const float* w_data = Tensor::data(weight);
                float* in_grad = Tensor::mutable_grad(input);
                for (int n = 0; n < N; ++n) {
                    for (int oc = 0; oc < C_out; ++oc) {
                        for (int oh = 0; oh < H_out; ++oh) {
                            for (int ow = 0; ow < W_out; ++ow) {
                                float g = out_grad[n * out_chw + oc * out_hw + oh * W_out + ow];
                                for (int ic = 0; ic < C_in; ++ic) {
                                    for (int kh = 0; kh < K_h; ++kh) {
                                        for (int kw = 0; kw < K_w; ++kw) {
                                            int ih = oh * stride - padding + kh;
                                            int iw = ow * stride - padding + kw;
                                            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                                int w_idx  = oc * w_chw + ic * w_hw + kh * K_w + kw;
                                                int in_idx = n * in_chw + ic * in_hw + ih * W_in + iw;
                                                in_grad[in_idx] += g * w_data[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }, input, weight, bias);
    }

    return out;
}
Tensor conv2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   int stride, int padding)
{
    const int N    = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H_in = static_cast<int>(input.shape()[2]);
    const int W_in = static_cast<int>(input.shape()[3]);

    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_c   = static_cast<int>(weight.shape()[1]);
    const int K_h   = static_cast<int>(weight.shape()[2]);
    const int K_w   = static_cast<int>(weight.shape()[3]);

    if (C_in != K_c || static_cast<int>(bias.shape()[0]) != C_out) {
        throw std::invalid_argument("conv2d_cuda: channel mismatch");
    }
    if (stride <= 0 || padding < 0) {
        throw std::invalid_argument("conv2d_cuda: invalid stride/padding");
    }

    const int H_out = (H_in + 2 * padding - K_h) / stride + 1;
    const int W_out = (W_in + 2 * padding - K_w) / stride + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("conv2d_cuda: output size is non-positive");
    }

    // ========== 前向核函数启动 ==========
    const bool need_grad = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    Tensor out({N, C_out, H_out, W_out}, need_grad,input.device());

    const float* d_input  = input.data();
    const float* d_weight = weight.data();
    const float* d_bias   = bias.data();
    float* d_output       = out.mutable_data();

    constexpr int block_size = 256;
    const int total_out = N * C_out * H_out * W_out;
    const int grid_size = (total_out + block_size - 1) / block_size;

    conv2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_weight, d_bias, d_output,
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride, padding,
        H_out, W_out
    );

    // 建议添加：CUDA错误检查（调试阶段非常有用）
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // ========== 反向传播注册 ==========
    if (need_grad) {
        out.set_backward_fn([
            bias_req_grad = bias.requires_grad(),
            weight_req_grad = weight.requires_grad(),
            input_req_grad = input.requires_grad(),
            stride, padding,
            N, C_in, H_in, W_in,
            C_out, K_h, K_w,
            H_out, W_out
        ](const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            const Tensor::NodePtr input_node  = out_inputs[0];
            const Tensor::NodePtr weight_node = out_inputs[1];
            const Tensor::NodePtr bias_node   = out_inputs[2];
            constexpr int block_size = 256;

            // 1. 偏置梯度
            if (bias_req_grad) {
                float* d_bias_grad = Tensor::mutable_grad(bias_node);
                const int grid = (C_out + block_size - 1) / block_size;
                conv2d_bias_backward_kernel<<<grid, block_size>>>(
                    out_grad, d_bias_grad, N, C_out, H_out, W_out
                );
            }

            // 2. 权重梯度
            if (weight_req_grad) {
                const float* d_input = Tensor::data(input_node);
                float* d_w_grad = Tensor::mutable_grad(weight_node);
                const int total_w = C_out * C_in * K_h * K_w;
                const int grid = (total_w + block_size - 1) / block_size;
                conv2d_weight_backward_kernel<<<grid, block_size>>>(
                    out_grad, d_input, d_w_grad,
                    N, C_in, H_in, W_in,
                    C_out, K_h, K_w,
                    stride, padding,
                    H_out, W_out
                );
            }

            // 3. 输入梯度
            if (input_req_grad) {
                const float* d_weight = Tensor::data(weight_node);
                float* d_in_grad = Tensor::mutable_grad(input_node);
                const int total_out = N * C_out * H_out * W_out;
                const int grid = (total_out + block_size - 1) / block_size;
                conv2d_input_backward_kernel<<<grid, block_size>>>(
                    out_grad, d_weight, d_in_grad,
                    N, C_in, H_in, W_in,
                    C_out, K_h, K_w,
                    stride, padding,
                    H_out, W_out
                );
            }
        }, input, weight, bias);
    }

    return out;
}

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride, int padding)
{
    // ========== 形状校验 ==========
    if (input.shape().size() != 4 || weight.shape().size() != 4 || bias.shape().size() != 1) {
        throw std::invalid_argument("conv2d: invalid tensor dimension");
    }
    if(input.device()==Device::kCPU&&weight.device()==Device::kCPU&&bias.device()==Device::kCPU){
        return conv2d_cpu(input,weight,bias,stride,padding);
    }
    if(input.device()==Device::kCUDA&&weight.device()==Device::kCUDA&&bias.device()==Device::kCUDA){
        return conv2d_cuda(input,weight,bias,stride,padding);
    }
    throw std::invalid_argument("conv2d: invalid device");
}


// Tensor max_pool2d(const Tensor& input, int kernel_size, int stride, int padding) {
//     if (input.shape().size() != 4) {
//         throw std::invalid_argument("max_pool2d: input must be 4D tensor");
//     }
//     if (kernel_size <= 0 || padding < 0) {
//         throw std::invalid_argument("max_pool2d: invalid kernel/padding");
//     }
//     // 步长默认等于核大小（标准下采样）
//     if (stride <= 0) stride = kernel_size;

//     int N = static_cast<int>(input.shape()[0]);
//     int C = static_cast<int>(input.shape()[1]);
//     int H_in = static_cast<int>(input.shape()[2]);
//     int W_in = static_cast<int>(input.shape()[3]);

//     int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
//     int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

//     Tensor out({N, C, H_out, W_out}, input.requires_grad());
//     const float* in_data = input.data();
//     float* out_data = out.mutable_data();

//     int in_hw = H_in * W_in;
//     int in_chw = C * in_hw;
//     int out_hw = H_out * W_out;
//     int out_chw = C * out_hw;

//     // 记录最大值位置，用于反向传播
//     std::vector<int> max_idx_h(out.numel());
//     std::vector<int> max_idx_w(out.numel());

//     for (int n = 0; n < N; ++n) {
//         for (int c = 0; c < C; ++c) {
//             for (int oh = 0; oh < H_out; ++oh) {
//                 for (int ow = 0; ow < W_out; ++ow) {
//                     float max_val = -INFINITY;
//                     int best_h = -1, best_w = -1;
//                     for (int kh = 0; kh < kernel_size; ++kh) {
//                         for (int kw = 0; kw < kernel_size; ++kw) {
//                             int ih = oh * stride - padding + kh;
//                             int iw = ow * stride - padding + kw;
//                             if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
//                                 int idx = n * in_chw + c * in_hw + ih * W_in + iw;
//                                 if (in_data[idx] > max_val) {
//                                     max_val = in_data[idx];
//                                     best_h = ih;
//                                     best_w = iw;
//                                 }
//                             }
//                         }
//                     }
//                     int out_idx = n * out_chw + c * out_hw + oh * W_out + ow;
//                     out_data[out_idx] = max_val;
//                     max_idx_h[out_idx] = best_h;
//                     max_idx_w[out_idx] = best_w;
//                 }
//             }
//         }
//     }

//     if (input.requires_grad()) {
//         out.set_backward_fn([max_idx_h, max_idx_w, stride, padding, kernel_size,
//         N,C,H_in,W_in,H_out,W_out,in_hw,in_chw,out_hw,out_chw]
//             (const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
//             Tensor::NodePtr input=out_inputs[0];
//             float* in_grad = Tensor::mutable_grad(input);

//             // int N = static_cast<int>(input.shape()[0]);
//             // int C = static_cast<int>(input.shape()[1]);
//             // int H_in = static_cast<int>(input.shape()[2]);
//             // int W_in = static_cast<int>(input.shape()[3]);
//             // int H_out = static_cast<int>(out.shape()[2]);
//             // int W_out = static_cast<int>(out.shape()[3]);

//             // int in_hw = H_in * W_in;
//             // int in_chw = C * in_hw;
//             // int out_hw = H_out * W_out;
//             // int out_chw = C * out_hw;

//             // 仅将梯度回传到最大值对应的位置
//             for (int n = 0; n < N; ++n) {
//                 for (int c = 0; c < C; ++c) {
//                     for (int oh = 0; oh < H_out; ++oh) {
//                         for (int ow = 0; ow < W_out; ++ow) {
//                             int out_idx = n * out_chw + c * out_hw + oh * W_out + ow;
//                             int ih = max_idx_h[out_idx];
//                             int iw = max_idx_w[out_idx];
//                             if (ih >= 0 && iw >= 0) {
//                                 int in_idx = n * in_chw + c * in_hw + ih * W_in + iw;
//                                 in_grad[in_idx] += out_grad[out_idx];
//                             }
//                         }
//                     }
//                 }
//             }
//         }, input);
//     }

//     return out;
// }
Tensor flatten_cpu(const Tensor& x) {
    int64_t batch = x.shape()[0];
    int64_t feat_dim = static_cast<int64_t>(x.numel()) / batch;

    Tensor out({batch, feat_dim}, x.requires_grad(),x.device());
    // 内存连续，直接拷贝数据
    std::copy(x.data(), x.data() + x.numel(), out.mutable_data());

    if (x.requires_grad()) {
        out.set_backward_fn([](const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            Tensor::NodePtr x=out_inputs[0];
            // 梯度直接按内存连续拷贝回原形状
            std::copy(out_grad, out_grad + out_numel, Tensor::mutable_grad(x));
        }, x);
    }

    return out;
}
Tensor flatten_cuda(const Tensor& x) {
    int64_t batch = x.shape()[0];
    int64_t feat_dim = static_cast<int64_t>(x.numel()) / batch;

    Tensor out({batch, feat_dim}, x.requires_grad(),x.device());
    // 内存连续，直接拷贝数据
    cudaMemcpy(out.mutable_data(),x.data(),x.numel()*sizeof(float),cudaMemcpyDeviceToDevice);

    if (x.requires_grad()) {
        out.set_backward_fn([](const float* out_grad, size_t out_numel, const std::vector<Tensor::NodePtr>& out_inputs) {
            Tensor::NodePtr x=out_inputs[0];
            // 梯度直接按内存连续拷贝回原形状
            cudaMemcpy(Tensor::mutable_grad(x),out_grad,out_numel*sizeof(float),cudaMemcpyDeviceToDevice);
        }, x);
    }

    return out;
}
Tensor flatten(const Tensor& x) {
    if (x.shape().empty()) {
        throw std::invalid_argument("flatten: input tensor is empty");
    }

    if(x.device()==Device::kCPU){
        return flatten_cpu(x);
    }else if(x.device()==Device::kCUDA){
        return flatten_cuda(x);
    }

    throw std::invalid_argument("flatten: invalid Device");
}
}  // namespace simpledl