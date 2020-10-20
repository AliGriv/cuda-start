//
// Created by AliGriv on 2020-10-19.
//
#ifndef _FUNCS_CUDA_H
#define _FUNCS_CUDA_H
/* Gpu code for adding two vecs */
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <cmath>
__global__ void VectorAdd_Kernel(const double *a, const double *b, double *c, const int n);
void VectorAdd_GPU(const double *h_a, const double *h_b, double *h_c, const int n);

#endif