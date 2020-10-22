#include <iostream>
#include "funcs_cuda.cuh"
// Number of threads in each thread block
const int blockSize = 128;

void initializeCUDA(int argc, char **argv, int &devID)
{
    findCudaDevice(argc, (const char **)argv);
//    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
//    cudaError_t error;
//    devID = 0;
//
//    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
//    {
//    devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
//    error = cudaSetDevice(devID);
//
//    if (error != cudaSuccess)
//    {
//    printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
//    exit(EXIT_FAILURE);
//    }
//    }
//
//    // get number of SMs on this GPU
//    error = cudaGetDevice(&devID);
//
//    if (error != cudaSuccess)
//    {
//    printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
//    exit(EXIT_FAILURE);
//    }
//
//    cudaDeviceProp deviceProp;
//
//    error = cudaGetDeviceProperties(&deviceProp, devID);
//
//    if (error != cudaSuccess)
//    {
//    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
//    exit(EXIT_FAILURE);
//    }
//
//    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

}
__global__ void VectorAdd_Kernel(const double *a, const double *b, double *c, const int n) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (id < n) {
        c[id] = a[id] + b[id];
    }

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

void VectorsClass::VectorAdd_GPU_InClass(double *h_a, double *h_b, double *h_c, int n) {
    // Copy host vectors to device
    cudaMemcpy( this->d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( this->d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread block
//    int blockSize = 10000;

    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    VectorAdd_Kernel<<<gridSize, blockSize>>>(this->d_a, this->d_b, this->d_c, n);
    cudaThreadSynchronize();
    cudaMemcpy( h_c, this->d_c, bytes, cudaMemcpyDeviceToHost );
    // Release device memory

}
void VectorsClass::VectorAdd_GPU_InClass(double *h_c, int n) {

    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    VectorAdd_Kernel<<<gridSize, blockSize>>>(this->d_a, this->d_b, this->d_c, n);
    std::cout << "bytes is " << this->bytes << std::endl;
    checkCudaErrors(cudaMemcpy( h_c, this->d_c, this->bytes, cudaMemcpyDeviceToHost ));
//    for (int i {0}; i < 10; ++i) {
//        std::cout << h_c[i] << std::endl;
//    }
    // Release device memory
}