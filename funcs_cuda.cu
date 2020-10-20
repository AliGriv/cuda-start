

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
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread block
    int blockSize = 10000;

    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    VectorAdd_Kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}