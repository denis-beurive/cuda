#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <sys/time.h>

#define SIZE 102400
#define MOD 102399
#define STEP 128

/* ARRAY A INITIALIZER */
void init_a(int * a)
{
    int i;
    for(i=0; i<SIZE; i++) {
        a[i] = 1;
    }
}

/* ARRAY B INITIALIZER */
void init_b(int * b)
{
	int i, j;

	j=0;

	for(i=0; i<SIZE-1; i++) {
		b[j] = i;
		j = (j+STEP)%MOD;
	}	

    b[SIZE-1] = SIZE-1;
}

/* CHECKING A VALUES */
int check_a(int * a)
{
    int i;
    int correct = 1;
	for(i=0; i<SIZE; i++) {
		if(a[i] != (i+1)) {
			correct = 0;
		} 
	}	

    return correct;
}


// First version of the function (the original one).
__global__ void mykernel1(int * a, int * b, int N)
{
    for(int i =  threadIdx.x; i < N; i+=blockDim.x) {
        int v = b[i];
        a[v] = a[v] + v;
    }
}

// Second version of the function (question #3).
__global__ void mykernel2(int * a, int * b, int *count, int N)
{
    // blockIdx.x: position of the block within the grid.
    // blockDim.x: dimension of a block (relatively to the direction "x").
    // threadIdx.x: position of the thread relatively to the block.

    printf("blockIdx.x:%d * blockDim.x:%d + threadIdx.x:%d => %d\n",
           blockIdx.x,
           blockDim.x,
           threadIdx.x,
           blockIdx.x * blockDim.x + threadIdx.x);
    *count = *count + 1;

    int index =  blockIdx.x * blockDim.x + threadIdx.x;
    for(int i=index; i<N; i+=gridDim.x * blockDim.x) {
        int v = b[i];
        a[v] = a[v] + v;
    }

}

#define Q 2

int main(int argc, char * argv[])
{
    struct timeval start;
    struct timeval stop;

    int * a = (int *)malloc(sizeof(int)*SIZE);
	int * b = (int *)malloc(sizeof(int)*SIZE);
	int count = 0;

    init_a(a);
	init_b(b);

    /*  INSERT CUDA ALLOCATION AND COPY HERE */
    int * d_a, * d_b, *d_count;
    cudaMalloc(&d_a, sizeof(int)*SIZE);
    cudaMalloc(&d_b, sizeof(int)*SIZE);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_a, a, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

	dim3 nBlocks;
	dim3 nThperBlock;

	// We define 1024 threads in only one block.
	// This is not a good solution.
	if (1 == Q) {
        nBlocks.x = 1; // number of block in the grid.
        nThperBlock.x = 1024; // number of threads per bloc
	}
	else {
        nBlocks.x = 16; // number of blocks in the grid.
        nThperBlock.x = 1024; // number of threads per bloc
	}

	// Execute the "kernel" in the GPU.
    // nBlocks: number of block in the grid
    // nThperBlock: number of threads per bloc
    gettimeofday(&start, nullptr);
	if (1 == Q) {
        mykernel1<<< nBlocks , nThperBlock >>>(d_a, d_b, SIZE);
	} else {
        printf("nBlocks = %d\n", nBlocks.x);
        printf("nThperBlock = %d\n", nThperBlock.x);
        mykernel2<<< nBlocks , nThperBlock >>>(d_a, d_b, d_count, SIZE);
	}

	// The kernel executes asynchronously relatively to the CPU.
	// That is: (1) the CPU starts the kernel.
	//          (2) the kernel starts its execution.
	//          (3) but, before the kernel stops, the CPU continues its execution!
	// Thus, we need to synchronize the CPU and the kernel.
	// The function "cudaDeviceSynchronize" waits for the kernel to finish.
    cudaDeviceSynchronize();
    gettimeofday(&stop, nullptr);
    printf("Execution duration: %ld (s) %ld (us)\n", stop.tv_sec - start.tv_sec, stop.tv_usec - start.tv_usec);

	// Copy the result from the GPU to the RAM.
    // Note: cudaMemcpy(dst, src, count, kind)
    cudaMemcpy(a, d_a, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Total number of loops: %d\n", count);

	int correct = check_a(a);;
	
	if(0 == correct) {
		printf("\n\n ******************** \n ***/!\\ ERROR /!\\ *** \n ******************** \n\n");
	}
	else {
		printf("\n\n ******************** \n ***** SUCCESS! ***** \n ******************** \n\n");
	}

	return 1;
}




