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

class VectorsClass {
private:
    double *d_a;
    double *d_b;
    double *d_c;
    size_t bytes;
public:
    VectorsClass(int N) {
        bytes = N*sizeof(double);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
    }
    VectorsClass(const double *h_a, const double *h_b, int N) {
        bytes = N*sizeof(double);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cudaMemcpy( this->d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy( this->d_b, h_b, bytes, cudaMemcpyHostToDevice);
    }
    void VectorAdd_GPU_InClass(const double *h_a, const double *h_b, double *h_c, const int n);
    void VectorAdd_GPU_InClass(double *h_c, const int n);
    ~VectorsClass() {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
};

#endif