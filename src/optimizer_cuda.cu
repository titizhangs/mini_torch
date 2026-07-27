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
void SGD_step_cuda(std::vector<Tensor>&  params_,float lr_) {
    for (auto& param : params_) {
        float* data = param.mutable_data();
        const float* grad = param.grad();
        size_t n = param.numel();
        float tmp_data[n];
        float tmp_grad[n];
        CHECK_CUDA(cudaMemcpy(tmp_data, data,
                 n*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(tmp_grad, grad,
                 n*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            tmp_data[i] -= lr_ * tmp_grad[i];
        }
        CHECK_CUDA(cudaMemcpy(data, tmp_data,
                 n*sizeof(float), cudaMemcpyHostToDevice));
    }
}
}  // namespace simpledl