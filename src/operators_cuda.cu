#include "operators.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CHECK_CUDA(call)                                 \
    do {                                                 \
        cudaError_t err = call;                          \
        if (err != cudaSuccess) {                        \
            throw std::runtime_error(                    \
                std::string("CUDA Error: ") + cudaGetErrorString(err) \
            );                                           \
        }                                                \
    } while(0)

namespace simpledl {
namespace {

constexpr int kReduceBlockSize = 256;
constexpr int CUDA_BLOCK_SIZE = 256; 

// 归约核函数：分块求和
__global__ void sum_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ block_sums,
    int64_t numel
) {
    __shared__ float s_sum[kReduceBlockSize];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * kReduceBlockSize + tid;

    if (idx < numel) {
        s_sum[tid] = input[idx];
    } else {
        s_sum[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = kReduceBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = s_sum[0];
    }
}

// 反向广播核函数
__global__ void sum_backward_kernel(
    const float* out_grad,
    float* input_grad,
    int64_t numel
) {
    int64_t idx = blockIdx.x * CUDA_BLOCK_SIZE + threadIdx.x;
    if (idx < numel) {
        input_grad[idx] += out_grad[0];
    }
}

} // anonymous namespace

// 对外暴露：前向纯计算接口
void sum_forward_cuda(const float* input, int64_t numel, float* output) {
    int64_t grid_size = (numel + kReduceBlockSize - 1) / kReduceBlockSize;

    float* d_block_sums = nullptr;
    CHECK_CUDA(cudaMalloc(&d_block_sums, grid_size * sizeof(float)));

    sum_reduce_kernel<<<grid_size, kReduceBlockSize>>>(input, d_block_sums, numel);
    CHECK_CUDA(cudaGetLastError());

    if (grid_size > 1) {
        sum_reduce_kernel<<<1, kReduceBlockSize>>>(d_block_sums, d_block_sums, grid_size);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaMemcpy(output, d_block_sums, sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(d_block_sums));
}

// 对外暴露：反向纯计算接口
void sum_backward_cuda(const float* out_grad, float* input_grad, int64_t numel) {
    int64_t grid_size = (numel + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    sum_backward_kernel<<<grid_size, CUDA_BLOCK_SIZE>>>(out_grad, input_grad, numel);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void scale_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ block_scale,
    int64_t numel,
    float scale_val
) {
    int64_t idx = blockIdx.x * CUDA_BLOCK_SIZE + threadIdx.x;
    if(idx<numel){
        block_scale[idx]=input[idx]*scale_val;
    }
}
// 反向广播核函数
__global__ void scale_backward_kernel(
    const float* __restrict__ out_grad,
    float* __restrict__ input_grad,
    int64_t numel,
    float scale_val
) {
    int64_t idx = blockIdx.x * CUDA_BLOCK_SIZE + threadIdx.x;
    if (idx < numel) {
        input_grad[idx] += out_grad[idx] * scale_val;
    }
}
// 对外暴露：前向纯计算接口
void scale_forward_cuda(const float* input, float scale_val, int64_t numel, float* output) {
    int64_t grid_size = (numel + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    float* block_scale = nullptr;
    CHECK_CUDA(cudaMalloc(&block_scale, numel * sizeof(float)));

    scale_forward_kernel<<<grid_size, CUDA_BLOCK_SIZE>>>(input, block_scale, numel, scale_val);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(output, block_scale,numel * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(block_scale));
}


// 对外暴露：反向纯计算接口
void scale_backward_cuda(const float* out_grad, float* input_grad, int64_t numel, float scale_val) {
    int64_t grid_size = (numel + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    scale_backward_kernel<<<grid_size, CUDA_BLOCK_SIZE>>>(out_grad, input_grad, numel, scale_val);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void matmul_forward_kernel(
    const float* __restrict__ x_data,
    const float* __restrict__ w_data,
    float* __restrict__ out_data,
    int64_t M,
    int64_t K,
    int64_t N
) {
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row<M&&col<N){
        float sum = 0.0f;
        for(int64_t kk=0;kk<K;++kk){
            sum += x_data[row*K+kk]*w_data[kk*N+col];
        }
        out_data[row*N+col]=sum;
    }
}

__global__ void matmul_backward_kernel_x(
    const float* __restrict__ out_grad,
    float* __restrict__ x_grad,
    const float* __restrict__ weight_data,
    int64_t M,
    int64_t K,
    int64_t N
) {
    int64_t ii = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t kk = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(ii<M&&kk<K){
        float grid = 0.0f;
        for(int64_t jj=0;jj<N;++jj){
            grid += out_grad[ii*N+jj]*weight_data[kk*N+jj];
        }
        x_grad[ii*K+kk]=grid;
    }
}

__global__ void matmul_backward_kernel_weight(
    const float* __restrict__ out_grad,
    float* __restrict__ weight_grad,
    const float* __restrict__ x_data,
    int64_t M,
    int64_t K,
    int64_t N
) {
    int64_t kk = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t jj = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(kk<K&&jj<N){
        float grid = 0.0f;
        for(int64_t ii=0;ii<M;++ii){
            grid += x_data[ii*K+kk]*out_grad[ii*N+jj];
        }
        weight_grad[kk*N+jj]=grid;
    }
}

void matmul_forward_cuda(const float* x_data,const float* w_data,float* out_data,int64_t M,int64_t K,int64_t N){
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    matmul_forward_kernel<<<gridDim, blockDim>>>(x_data,w_data,out_data,M, K, N);
    CHECK_CUDA(cudaGetLastError());
}
void matmul_backward_cuda(const float* out_grad,bool x_req_grad,bool weight_req_grad
    ,float* x_grad,float* weight_grad,const float* x_data,const float* weight_data,int64_t M,int64_t K,int64_t N){
    dim3 blockDim(16, 16);
    if(x_req_grad){
        dim3 gridDim(
        (K + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
        );
        matmul_backward_kernel_x<<<gridDim, blockDim>>>(out_grad,x_grad,weight_data,M,K,N);
        CHECK_CUDA(cudaGetLastError());
    }
    if(weight_req_grad){
        dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (K + blockDim.y - 1) / blockDim.y
        );
        matmul_backward_kernel_weight<<<gridDim, blockDim>>>(out_grad,weight_grad,x_data,M,K,N);
        CHECK_CUDA(cudaGetLastError());
    }
}
__global__ void bias_add_forward_kernel_x(
    float* __restrict__ out_data,
    const float* __restrict__ x_data,
    const float* __restrict__ b_data,
    int64_t M,
    int64_t N
)   {
    int64_t row=blockIdx.y*blockDim.y + threadIdx.y;
    int64_t col=blockIdx.x*blockDim.x + threadIdx.x;
    if(row<M&&col<N){
        out_data[row*N+col]=x_data[row*N+col]+b_data[row];
    }
}

__global__ void bias_add_backward_kernel_x(
float* __restrict__ x_grad,
const float* __restrict__ out_grad,
int64_t M,
int64_t N
)   {
    int64_t row=blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col=blockIdx.x * blockDim.x + threadIdx.x;
    if(row<M&&col<N){
        x_grad[row*N+col] += out_grad[row*N+col];
    }
}

__global__ void bias_add_backward_kernel_bias(
float* __restrict__ bias_grad,
const float* __restrict__ out_grad,
int64_t M,
int64_t N){
    int64_t row=blockIdx.x * blockDim.x + threadIdx.x;
    if(row<M){
        for(int64_t ii=0;ii<N;++ii){
            bias_grad[row] += out_grad[row*N+ii];
        }
    }
}

void bias_add_forward_cuda(float* out_data,const float* x_data,const float* b_data,int64_t M,int64_t N){
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    bias_add_forward_kernel_x<<<gridDim,blockDim>>>(out_data,x_data,b_data,M,N);
    CHECK_CUDA(cudaGetLastError());
}

void bias_add_backward_cuda_x(float* x_grad,const float* out_grad,int64_t M,int64_t N){
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    bias_add_backward_kernel_x<<<gridDim,blockDim>>>(x_grad,out_grad,M,N);
    CHECK_CUDA(cudaGetLastError());
}
void bias_add_backward_cuda_bias(float* bias_grad,const float* out_grad,int64_t M, int64_t N){
    bias_add_backward_kernel_bias<<<(M+CUDA_BLOCK_SIZE-1)/CUDA_BLOCK_SIZE,CUDA_BLOCK_SIZE>>>(bias_grad,out_grad,M,N);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void relu_forward_kernel(float* __restrict__ out_data,const float* __restrict__  x_data,int64_t numel){
    int64_t row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<numel){
        if(x_data[row]>0.0f){
            out_data[row]=x_data[row];
        }else{
            out_data[row]=0.0f;
        }
    }
}
__global__ void relu_backward_kernel(float* __restrict__ x_grad,const float* __restrict__ out_grad,
    const float* __restrict__ x_data,int64_t out_numel){
    int64_t row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<out_numel){
        if(x_data[row]>0.0f){
            x_grad[row] += out_grad[row];
        }
    }
}

void relu_forward_cuda(float* out_data,const float* x_data,int64_t numel){
    relu_forward_kernel<<<(numel+CUDA_BLOCK_SIZE-1)/CUDA_BLOCK_SIZE,CUDA_BLOCK_SIZE>>>(out_data,x_data,numel);
}

void relu_backward_cuda(float* x_grad,const float* out_grad,const float* x_data,int64_t out_numel){
    relu_backward_kernel<<<(out_numel+CUDA_BLOCK_SIZE-1)/CUDA_BLOCK_SIZE,CUDA_BLOCK_SIZE>>>(x_grad,out_grad,x_data,out_numel);
}


// =============================================================
// 1. 前向传播核函数
// 并行策略：每个线程对应1个输出元素 (n, oc, oh, ow)
// 内层串行完成输入通道、卷积核高宽的乘加运算
// =============================================================
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int padding,
    int H_out, int W_out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * H_out * W_out;
    if (idx >= total_out) return;

    // 从线性索引反解出 (n, oc, oh, ow)
    const int n = idx / (C_out * H_out * W_out);
    int rem = idx % (C_out * H_out * W_out);
    const int oc = rem / (H_out * W_out);
    rem %= H_out * W_out;
    const int oh = rem / W_out;
    const int ow = rem % W_out;

    float sum = bias[oc];
    const int in_n_base = n * C_in * H_in * W_in;
    const int w_oc_base = oc * C_in * K_h * K_w;

    // 串行遍历输入通道与卷积核空间
    for (int ic = 0; ic < C_in; ++ic) {
        const int in_ic_base = in_n_base + ic * H_in * W_in;
        const int w_ic_base = w_oc_base + ic * K_h * K_w;
        for (int kh = 0; kh < K_h; ++kh) {
            const int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= H_in) continue;  // 越界直接跳过（等效零填充）
            for (int kw = 0; kw < K_w; ++kw) {
                const int iw = ow * stride - padding + kw;
                if (iw < 0 || iw >= W_in) continue;
                const int in_idx = in_ic_base + ih * W_in + iw;
                const int w_idx  = w_ic_base + kh * K_w + kw;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    output[idx] = sum;
}

// =============================================================
// 2. 偏置梯度核函数
// 并行策略：每个线程对应1个输出通道，串行累加所有batch与空间位置的梯度
// =============================================================
__global__ void conv2d_bias_backward_kernel(
    const float* __restrict__ out_grad,
    float* __restrict__ bias_grad,
    int N, int C_out, int H_out, int W_out)
{
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= C_out) return;

    float sum = 0.0f;
    const int out_chw = C_out * H_out * W_out;
    const int out_hw  = H_out * W_out;

    for (int n = 0; n < N; ++n) {
        const int base = n * out_chw + oc * out_hw;
        // 空间维度连续内存直接遍历，提升访存效率
        for (int i = 0; i < out_hw; ++i) {
            sum += out_grad[base + i];
        }
    }
    bias_grad[oc] += sum;
}

// =============================================================
// 3. 权重梯度核函数
// 并行策略：每个线程对应1个权重元素 (oc, ic, kh, kw)
// 串行累加所有batch与输出空间位置的梯度
// =============================================================
__global__ void conv2d_weight_backward_kernel(
    const float* __restrict__ out_grad,
    const float* __restrict__ input,
    float* __restrict__ weight_grad,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int padding,
    int H_out, int W_out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_w = C_out * C_in * K_h * K_w;
    if (idx >= total_w) return;

    // 从线性索引反解出 (oc, ic, kh, kw)
    const int oc = idx / (C_in * K_h * K_w);
    int rem = idx % (C_in * K_h * K_w);
    const int ic = rem / (K_h * K_w);
    rem %= K_h * K_w;
    const int kh = rem / K_w;
    const int kw = rem % K_w;

    float sum = 0.0f;
    const int out_chw = C_out * H_out * W_out;
    const int out_hw  = H_out * W_out;
    const int in_chw  = C_in * H_in * W_in;
    const int in_hw   = H_in * W_in;

    for (int n = 0; n < N; ++n) {
        const int out_n_base = n * out_chw + oc * out_hw;
        const int in_n_base  = n * in_chw + ic * in_hw;
        for (int oh = 0; oh < H_out; ++oh) {
            const int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= H_in) continue;
            for (int ow = 0; ow < W_out; ++ow) {
                const int iw = ow * stride - padding + kw;
                if (iw < 0 || iw >= W_in) continue;
                const int out_idx = out_n_base + oh * W_out + ow;
                const int in_idx  = in_n_base + ih * W_in + iw;
                sum += out_grad[out_idx] * input[in_idx];
            }
        }
    }

    weight_grad[idx] += sum;
}

// =============================================================
// 4. 输入梯度核函数
// 并行策略：每个线程对应1个输出梯度元素，将梯度散射回输入位置
// 使用 atomicAdd 保证多线程写同一输入位置的并发安全
// =============================================================
__global__ void conv2d_input_backward_kernel(
    const float* __restrict__ out_grad,
    const float* __restrict__ weight,
    float* __restrict__ input_grad,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int padding,
    int H_out, int W_out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * H_out * W_out;
    if (idx >= total_out) return;

    // 从线性索引反解出 (n, oc, oh, ow)
    const int n = idx / (C_out * H_out * W_out);
    int rem = idx % (C_out * H_out * W_out);
    const int oc = rem / (H_out * W_out);
    rem %= H_out * W_out;
    const int oh = rem / W_out;
    const int ow = rem % W_out;

    const float g = out_grad[idx];
    const int in_n_base = n * C_in * H_in * W_in;
    const int w_oc_base = oc * C_in * K_h * K_w;

    for (int ic = 0; ic < C_in; ++ic) {
        const int in_ic_base = in_n_base + ic * H_in * W_in;
        const int w_ic_base  = w_oc_base + ic * K_h * K_w;
        for (int kh = 0; kh < K_h; ++kh) {
            const int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= H_in) continue;
            for (int kw = 0; kw < K_w; ++kw) {
                const int iw = ow * stride - padding + kw;
                if (iw < 0 || iw >= W_in) continue;
                const int in_idx = in_ic_base + ih * W_in + iw;
                const int w_idx  = w_ic_base + kh * K_w + kw;
                atomicAdd(&input_grad[in_idx], g * weight[w_idx]);
            }
        }
    }
}


} // namespace simpledl
