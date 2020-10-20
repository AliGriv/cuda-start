
#include "funcs_cuda.cuh"
// Number of threads in each thread block
const int blockSize = 16384;


__global__ void VectorAdd_Kernel(const double *a, const double *b, double *c, const int n) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

void VectorAdd_GPU(const double *h_a, const double *h_b, double *h_c, const int n) {

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
    // Device input vectors
    double *dd_a;
    double *dd_b;
    //Device output vector
    double *dd_c;
    // Allocate memory for each vector on GPU
    cudaMalloc(&dd_a, bytes);
    cudaMalloc(&dd_b, bytes);
    cudaMalloc(&dd_c, bytes);
    // Copy host vectors to device
    cudaMemcpy( dd_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( dd_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread block
//    int blockSize = 10000;

    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    VectorAdd_Kernel<<<gridSize, blockSize>>>(dd_a, dd_b, dd_c, n);
    cudaMemcpy( h_c, dd_c, bytes, cudaMemcpyDeviceToHost );
    // Release device memory
    cudaFree(dd_a);
    cudaFree(dd_b);
    cudaFree(dd_c);
}

void VectorsClass::VectorAdd_GPU_InClass(const double *h_a, const double *h_b, double *h_c, const int n) {
    // Copy host vectors to device
    cudaMemcpy( this->d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( this->d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread block
//    int blockSize = 10000;

    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    VectorAdd_Kernel<<<gridSize, blockSize>>>(this->d_a, this->d_b, this->d_c, n);
    cudaMemcpy( h_c, this->d_c, bytes, cudaMemcpyDeviceToHost );
    // Release device memory

}
void VectorsClass::VectorAdd_GPU_InClass(double *h_c, const int n) {

    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    VectorAdd_Kernel<<<gridSize, blockSize>>>(this->d_a, this->d_b, this->d_c, n);
    cudaMemcpy( h_c, this->d_c, bytes, cudaMemcpyDeviceToHost );
    // Release device memory

}