// vector_add.cu

#include <iostream>
#include <cuda.h>
#include <cstdlib>   // for rand()
using namespace std;

#define BLOCK_SIZE 256  // number of threads per block (typical value)

__global__ void vector_add(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void fill_array(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;  // random numbers between 0 and 99
    }
}

void add_cpu(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void print_array(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main() {
    int n;
    cout << "Enter size of vectors: ";
    cin >> n;

    int *a, *b, *c; // host pointers
    int *d_a, *d_b, *d_c; // device pointers

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&d_a, n * sizeof(int));
    cudaMallocManaged(&d_b, n * sizeof(int));
    cudaMallocManaged(&d_c, n * sizeof(int));

    // Fill input vectors
    fill_array(d_a, n);
    fill_array(d_b, n);

    // Print input vectors
    cout << "Array 1: ";
    print_array(d_a, n);
    cout << "Array 2: ";
    print_array(d_b, n);

    // GPU Execution
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();  // wait for GPU to finish

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    cout << "\nGPU Result: ";
    print_array(d_c, n);
    cout << "GPU Time: " << milliseconds << " ms" << endl;

    // CPU Execution
    int *cpu_result = new int[n];

    clock_t cpu_start = clock();
    add_cpu(d_a, d_b, cpu_result, n);
    clock_t cpu_end = clock();

    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    cout << "\nCPU Result: ";
    print_array(cpu_result, n);
    cout << "CPU Time: " << cpu_time << " ms" << endl;

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] cpu_result;

    return 0;
}
