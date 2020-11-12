#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define N 256
#define TPB 256

/* After checking in many different forums, we have discovered that for some reason
	Visual Studio Intellisense does not recognize the <<< >>> nomenclature, so we
	have been forced to implement a macro to change that for KERNEL_ARGS2 and so
	as it can be seen in the following lines. 

	We know that the right way is to directly use the <<< >>> but we had no option.
	Hope you understand :)
*/

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif




__global__ void kernelPrint(int len)
{
	// We calculate the id with the general way although is just one block of threads
	// and could be easily obtained with threadIdx.x
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	// Then, we print the thread
	printf("Hello world! My threadId is %d.\n", i);

}

int main()
{ 
    
	printf("\n Let´s go printing! \n");

	// Launch kernel to print hello worlds with Ids
	kernelPrint KERNEL_ARGS2(N/TPB,TPB)(N);

	// We wait for the GPU
	cudaDeviceSynchronize();
	

	printf("\n Printing done! \n");
  
	return 0;
}