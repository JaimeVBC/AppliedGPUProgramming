#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#define SEED     921

#define TPB 32

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif


__global__ void kernelMonteCarlo(curandState *states, int *d_count, int num_iter_per_thread, int num_threads)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < num_threads)
    {
        double x, y, z;

        int seed = id; // different seed per thread
        curand_init(seed, id, 0, &states[id]); // Initialize CURAND
        int cont = 0;
        for (int i = 0; i < num_iter_per_thread; i++)
        {
            x = curand_uniform(&states[id]);
            y = curand_uniform(&states[id]);
            z = sqrt(x * x + y * y);
            if (z <= 1.0)
            {
                cont++;
            }
        }
        d_count[id] = cont;
    }
}


int main(int argc, char* argv[])
{
    printf("\nLet´s run bonus exercise! \n\n");
    int NUM_ITER = 16384*16;
    float pi = 0.0f;
    int num_iter_per_thread = 32;
    int num_blocks = (NUM_ITER + TPB - 1)/TPB;
    int num_threads = TPB * num_blocks / num_iter_per_thread;
    
    
    printf("\nNUM_ITER:  %d ", NUM_ITER);
    printf("\nNumIterPerThread:  %d ", num_iter_per_thread);
    printf("\nNumThreads:  %d ", num_threads);
    printf("\nThreads per block:  %d ", TPB);
    printf("\nNumber of thread blocks:  %d \n\n", num_blocks);

    srand(SEED); // Important: Multiply SEED by "rank" when you introduce MPI!

    curandState* dev_random;
    cudaMalloc((void**)&dev_random, num_threads * sizeof(curandState));

    int* d_counts; 
    cudaMalloc(&d_counts, num_threads * sizeof(int));

    int* counts;
    counts = (int*) malloc(num_threads * sizeof(int));
    if (counts == NULL) { printf("ERROR! Failure when allocating dynamic memory"); return 1; }

    kernelMonteCarlo KERNEL_ARGS2((num_threads+TPB-1)/TPB, TPB)(dev_random, d_counts, num_iter_per_thread,num_threads);
    cudaDeviceSynchronize();
    cudaMemcpy(counts, d_counts, num_threads * sizeof(int), cudaMemcpyDeviceToHost);


    printf("GPU has finished!\n\n");

    

    int accumulate_count = 0;
    for (int i = 0; i < num_threads; i++)
    {
        /*printf("i vale: %d\n", i);
        printf("counts[i] vale: %d\n", counts[i]);
        printf("accumulate_count vale: %d\n\n", accumulate_count);*/
        accumulate_count += counts[i];
    }
    pi = (float) accumulate_count / NUM_ITER * 4;

    printf("The approximate result of PI is: %lf\n", pi);

    free(counts);
    cudaFree(d_counts);
    cudaFree(dev_random);

    printf("\nBonus exercise completed! \n\n");

    return 0;
}
