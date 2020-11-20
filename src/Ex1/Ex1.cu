#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Compilation:
//      nvcc Ex1.cu -o Ex1.exe


// __global__ => this function executes on the GPU.
// Please note that it also could be: __device__.
// This is this only code that executes on the GPU.

__global__ void kernel(double *a, double *b, double *c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (i < N) {
	    c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv)
{
    int N = 1000;
    int sz_in_bytes = N*sizeof(double);

    double *h_a, *h_b, *h_c; // "h" for "host" (allocated in RAM).
    double *d_a, *d_b, *d_c; // "d" for "device" (allocated in the GPU).

    // Allocate memory in RAM (that is, the "host"):
    // 3 arrays that contain N elements. Each element is a "double".
    h_a = (double*)malloc(sz_in_bytes);
    h_b = (double*)malloc(sz_in_bytes);
    h_c = (double*)malloc(sz_in_bytes);

    // Initiate values on h_a and h_b
    for(int i = 0 ; i < N ; i++) {
        h_a[i] = 1./(1.+i);
        h_b[i] = (i-1.)/(i+1.);
    }

    // Allocate memory in the GPU (that is, the "device").
    cudaMalloc((void**)&d_a, sz_in_bytes);
    cudaMalloc((void**)&d_b, sz_in_bytes);
    cudaMalloc((void**)&d_c, sz_in_bytes);

    // Copy the data from the RAM (host) to the GPU (device).
    // Note: cudaMemcpy(dst, src, count, kind)
    cudaMemcpy(d_a, h_a, sz_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz_in_bytes, cudaMemcpyHostToDevice);

    // Set 64*1*1 thread per blocks.
    // x: 64
    // y: 1
    // z: 1
    // Note: we statically initialize *structure**.
    dim3  dimBlock(64, 1, 1);

    // Set (N + dimBlock.x - 1)/dimBlock.x * 1 * 1 blocs.
    // If N=1000: (N + dimBlock.x - 1)/dimBlock.x => 16 blocks
    //            (1000 + 64 - 1) / 64 = 16
    //            (1000 + 64 - 1) % 64 = 39
    //            => There are more threads that elements in the array.
    // Note: dimBlock.x = 64.
    // Note: we statically initialize *structure**.
    dim3  dimGrid((N + dimBlock.x - 1)/dimBlock.x, 1, 1);

    // Thus, we have 64*16 = 1024 threads.

    // Run the "kernel" (in the GPU).
    // dimGrid: number of block in the greed => 16
    // dimBlock: number of threads per bloc => 64
    kernel<<<dimGrid , dimBlock>>>(d_a, d_b, d_c, N);

    // Result is pointed by d_c on device.
    // Copy this result on host (result pointed by h_c on host)
    cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost);

    // freeing on device 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
