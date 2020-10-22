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
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_string.h>
void initializeCUDA(int argc, char **argv, int &devID);


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
        checkCudaErrors(cudaMalloc(&d_a, bytes));
        checkCudaErrors(cudaMalloc(&d_b, bytes));
        checkCudaErrors(cudaMalloc(&d_c, bytes));
    }
    VectorsClass(double *h_a, double *h_b, int N) {
        bytes = N*sizeof(double);
        checkCudaErrors(cudaMalloc(&d_a, bytes));
        checkCudaErrors(cudaMalloc(&d_b, bytes));
        checkCudaErrors(cudaMalloc(&d_c, bytes));
        checkCudaErrors(cudaMemcpy( this->d_a, h_a, bytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy( this->d_b, h_b, bytes, cudaMemcpyHostToDevice));
    }
    void VectorAdd_GPU_InClass(double *h_a, double *h_b, double *h_c, int n);
    void VectorAdd_GPU_InClass(double *h_c, const int n);
    ~VectorsClass() {
        checkCudaErrors(cudaFree(d_a));
        checkCudaErrors(cudaFree(d_b));
        checkCudaErrors(cudaFree(d_c));
    }
};


#endif