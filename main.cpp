#include <iostream>
#include <stdio.h>
#include <chrono>
#include "funcs.h"
#include "funcs_cuda.cuh"

const int N = 10000000;


int main() {

    double *a = new double[N];
    double *b = new double[N];
    double *c = new double[N];
    for (int i {0}; i < N; ++i) {
        a[i] = (double) 10.0*rand()/RAND_MAX;
        b[i] = (double) 10.0*rand()/RAND_MAX;
    }

    std::cout << "vectors defined" << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();
    VectorAdd_CPU(a, b, c, N);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(end - begin);
    std::cout << "It took " << elapsed.count() << " seconds to compute on CPU!" << std::endl;


    begin = std::chrono::high_resolution_clock::now();
    VectorAdd_GPU(a, b, c, N);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(end - begin);
    std::cout << "It took " << elapsed.count() << " seconds to compute on GPU!" << std::endl;

    VectorsClass vecs1(N);
    begin = std::chrono::high_resolution_clock::now();
    vecs1.VectorAdd_GPU_InClass(a, b, c, N);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(end - begin);
    std::cout << "It took " << elapsed.count() << " seconds to compute on GPU with the class (excluding cudaMalloc time)!" << std::endl;


    VectorsClass vecs2(a,b,N);
    begin = std::chrono::high_resolution_clock::now();
    vecs2.VectorAdd_GPU_InClass(c, N);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(end - begin);
    std::cout << "It took " << elapsed.count() << " seconds to compute on GPU with the class (excluding cudaMalloc and cudaMemcpy time)!" << std::endl;
    delete [] a;
    delete [] b;
    delete [] c;
    return 0;
}
