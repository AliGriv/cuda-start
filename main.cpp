#include <iostream>
#include <stdio.h>
#include <chrono>
const int N = 10000000;

/* cpu code for adding two vectors*/
void VectorAdd_CPU(const double *a, const double *b, double *c, const int size) {
    for (int i {0}; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}
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

    delete [] a;
    delete [] b;
    delete [] c;
    return 0;
}
