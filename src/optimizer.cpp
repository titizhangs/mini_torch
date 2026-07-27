#include "optimizer.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
namespace simpledl {
void SGD_step_cuda(std::vector<Tensor>& params_,float lr_);
SGD::SGD(std::vector<Tensor> params, float lr)
    : params_(std::move(params)), lr_(lr) {}

void SGD::zero_grad() {
    for (Tensor& param : params_) {
        param.fill_grad_ptr(0.0f);
    }
}

void SGD::step() {
    if(params_[0].device()==Device::kCPU){
        for (auto& param : params_) {
            float* data = param.mutable_data();
            const float* grad = param.grad();
            size_t n = param.numel();
            for (size_t i = 0; i < n; ++i) {
                data[i] -= lr_ * grad[i];
            }
        }
    }else{
        SGD_step_cuda(params_,lr_);
    }
}
void SGD::show_params(){
    for (auto& param : params_) {
        float* data = param.mutable_data();
        const float* grad = param.grad();
        size_t n = param.numel();
        for (size_t i = 0; i < n; ++i) {
            printf("%f;",data[i]);
        }
        std::cout<<std::endl<<"=============="<<std::endl;
    }
}
}  // namespace simpledl