// utils.h
#pragma once
#include "Tensor.h"

void he_normal(Tensor& tensor, int fan_in);

void he_uniform(Tensor& tensor, int fan_in);
