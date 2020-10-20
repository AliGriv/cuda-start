//
// Created by AliGriv on 2020-10-18.
//

#ifndef CUDA_START_FUNCS_H
#define CUDA_START_FUNCS_H
//#include <cuda_runtime.h>
//#include <cuda.h>


/* cpu code for adding two vectors*/
void VectorAdd_CPU(const double *a, const double *b, double *c, const int size) {
    for (int i {0}; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}




#endif //CUDA_START_FUNCS_H
