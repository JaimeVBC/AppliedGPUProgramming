/*#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

/* After checking in many different forums, we have discovered that for some reason
	Visual Studio Intellisense does not recognize the <<< >>> nomenclature, so we
	have been forced to implement a macro to change that for KERNEL_ARGS2 and so
	as it can be seen in the following lines.

	We know that the right way is to directly use the <<< >>> but we had no option.
	Hope you understand :)
*//*
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif



#define TPB 512
#define NUM_PARTICLES 20000000





__global__ void kernelSAXPY(int len, float a, float* d_x, float* d_y)
{
	// We calculate the id with the general way although is just one block of threads
	// and could be easily obtained with threadIdx.x
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < len)
		d_y[i] = d_x[i] * a + d_y[i];

}

void cpuSAXPY(int len, float a, float* x, float* y)
{

	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		y[i] = x[i] * a + y[i];
	}

}


int gettimeofdaypocho(struct timeval* tp, struct timezone* tzp)
{
	static const unsigned __int64 epoch = 116444736000000000;

	FILETIME    file_time;
	SYSTEMTIME  system_time;
	ULARGE_INTEGER ularge;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	ularge.LowPart = file_time.dwLowDateTime;
	ularge.HighPart = file_time.dwHighDateTime;

	tp->tv_sec = (long)((ularge.QuadPart - epoch) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);

	return 0;
}

int gettimeofday(struct timeval* tv, struct timezone* tz)
{
	static LONGLONG birthunixhnsec = 116444736000000000;  /*in units of 100 ns *//*

	FILETIME systemtime;
	GetSystemTimeAsFileTime(&systemtime);

	ULARGE_INTEGER utime;
	utime.LowPart = systemtime.dwLowDateTime;
	utime.HighPart = systemtime.dwHighDateTime;

	ULARGE_INTEGER birthunix;
	birthunix.LowPart = (DWORD)birthunixhnsec;
	birthunix.HighPart = birthunixhnsec >> 32;

	LONGLONG usecs;
	usecs = (LONGLONG)((utime.QuadPart - birthunix.QuadPart) / 10);

	tv->tv_sec = (long long)(usecs / 1000000);
	tv->tv_usec = (long long)(usecs % 1000000);

	return 0;
}


int mainExcercise2()
{
	printf("\nLet´s run exercise 2! \n\n");

	printf("\nARRAY SIZE:  %d ", NUM_PARTICLES);
	printf("\nThreads per block:  %d ", TPB);
	printf("\nNumber of thread blocks:  %d \n\n", (NUM_PARTICLES+TPB-1)/TPB);



	float const a = 1.5;
	float* d_x;
	float* d_y;

	cudaMalloc(&d_x, NUM_PARTICLES*sizeof(float));
	cudaMalloc(&d_y, NUM_PARTICLES*sizeof(float));

	if (d_x == NULL || d_y == NULL)
	{
		printf("\n\nERROR 1! fail when allocating cuda dynamic memory!\n\n");
		return 1;
	}

	float* x = (float*) malloc(NUM_PARTICLES * sizeof(float));
	float* y = (float*) malloc(NUM_PARTICLES * sizeof(float));

	if (x == NULL || y == NULL)
	{
		printf("\n\nERROR 2! fail when allocating cuda dynamic memory!\n\n");
		return 1;
	}

	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		x[i] = (float) i*3;
		y[i] = (float) i/2;
	}

	cudaMemcpy(d_x, x, NUM_PARTICLES*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, NUM_PARTICLES*sizeof(float), cudaMemcpyHostToDevice);

	// Let's measuare the time
	struct timeval tStart;
	struct timeval tEnd;
	
	printf("Computing GPU Saxpy...  \n");

	// We start the timer
	gettimeofday(&tStart, NULL);

	// Launch kernel to print hello worlds with Ids
	kernelSAXPY KERNEL_ARGS2((NUM_PARTICLES+TPB-1)/TPB, TPB)(NUM_PARTICLES,a,d_x,d_y);

	// We wait for the GPU
	cudaDeviceSynchronize();

	// We stop the timer...
	gettimeofday(&tEnd,NULL);

	// And finally print the timer
	printf("GPU SAXPY completed in %3.10f miliseconds \n", ((tEnd.tv_sec - tStart.tv_sec) * 1000000.0 + (tEnd.tv_usec - tStart.tv_usec)) / 1000.0);
	cudaMemcpy(y, d_y, NUM_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);



	// Now let's go with the CPU Saxpy
	printf("\nComputing CPU Saxpy... \n");
	
	// We restart the timer 
	gettimeofday(&tStart, NULL);
	
	// The CPU Saxpy is computed
	cpuSAXPY(NUM_PARTICLES,a,x,y);

	// And then we stop the timer again
	gettimeofday(&tEnd,NULL);

	// Finally, we print the difference between the start time and the finished time
	printf("CPU SAXPY completed in %3.10f miliseconds \n", ((tEnd.tv_sec - tStart.tv_sec) * 1000000.0 + (tEnd.tv_usec - tStart.tv_usec)) / 1000.0);

	// Free resources of the 4 arrays that were dynamically allocated
	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);

	printf("\nExcercise 2 completed! \n");

	return 0;
}


*/