#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define checkCudaErrors(x) printf("%s (%d)\n", cudaGetErrorString(x), __LINE__)


__global__ void kernel(double *a, double *b, double *c, int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];

    // Possible correction: let one thread do more than one calculation.
    // If there is less threads than elements to calculate, then some threads will
    // do 2 calculations (instead of one).
    if (i < N-640) {
        c[i+640] = a[i+640] + b[i+640];
    }
}

int main(int argc, char **argv)
{
    int N = 1000;
    int sz_in_bytes = N*sizeof(double);

    double *h_a, *h_b, *h_c;
    double *d_a, *d_b, *d_c;

    h_a = (double*)malloc(sz_in_bytes);
    h_b = (double*)malloc(sz_in_bytes);
    h_c = (double*)malloc(sz_in_bytes);

    // Initiate values on h_a and h_b
    for(int i = 0 ; i < N ; i++) {
        h_a[i] = 1./(1.+i);
        h_b[i] = (i-1.)/(i+1.);
    }


    checkCudaErrors(cudaMalloc((void**)&d_a, sz_in_bytes));
    // Correction:
    // checkCudaErrors(cudaMalloc((void**)&d_b, 0));
    checkCudaErrors(cudaMalloc((void**)&d_b, sz_in_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_c, sz_in_bytes));

    checkCudaErrors(cudaMemcpy(d_a, h_a, sz_in_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sz_in_bytes, cudaMemcpyHostToDevice));

    // 640 threads au total.
    // But we calculate 1000 values.
    // => error.
    // One correction is: use enough threads.
    //    dim3  dimBlock(64, 1, 1);
    //    dim3  dimGrid(10, 1, 1) => dim3  dimGrid(10, 1, 1);
    // Another correction is:
    //    Let a thread make more than one calculation (see function kernel()).
    dim3  dimBlock(64, 1, 1);
    dim3  dimGrid(16, 1, 1);

    kernel<<<dimGrid , dimBlock>>>(d_a, d_b, d_c, N);

    checkCudaErrors(cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Verifying
    double err = 0, norm = 0;
    for(int i = 0 ; i < N ; i++) {
        double err_loc = fabs(h_c[i] - (h_a[i]+h_b[i]));
        err  += err_loc;
        norm += fabs(h_c[i]);
    }

    if (err/norm < 1.e-16) {
	    printf("SUCCESS (Relative error : %.3e)\n", err/norm);
    } else {
	    printf("ERROR (Relative error : %.3e)\n", err/norm);
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

