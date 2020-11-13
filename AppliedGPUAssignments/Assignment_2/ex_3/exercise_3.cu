#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

#include "exercise_2.cu"

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

#define TPB 512
#define NUM_PARTICLES 10000000
#define NUM_ITERATIONS 1000

struct Particle
{
	float3 position;
	float3 velocity;
};


__global__ void kernelParticlesUpdate(int len, Particle* d_particleArray)
{
	// We calculate the id with the general way although is just one block of threads
	// and could be easily obtained with threadIdx.x
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float dt = 1;
	if (i < len)
	{
		// Velocity update
		d_particleArray[i].velocity.x *= 0.5 * i;
		d_particleArray[i].velocity.y *= 2 * i;
		d_particleArray[i].velocity.z *= 0.75 * i;

		// Position update 
		d_particleArray[i].position.x += d_particleArray[i].velocity.x * dt;
		d_particleArray[i].position.y += d_particleArray[i].velocity.y * dt;
		d_particleArray[i].position.z += d_particleArray[i].velocity.z * dt;
	}
}

void cpuParticlesUpdate(Particle* particleArray)
{
	int i;
	for (i = 0; i < NUM_PARTICLES; i++)
	{
		float dt = 1;

		// Velocity update
		particleArray[i].velocity.x *= 0.5 * i;
		particleArray[i].velocity.y *= 2 * i;
		particleArray[i].velocity.z *= 0.75 * i;

		// Position update 
		particleArray[i].position.x += particleArray[i].velocity.x * dt;
		particleArray[i].position.y += particleArray[i].velocity.y * dt;
		particleArray[i].position.z += particleArray[i].velocity.z * dt;
	}

}


int gettimeofday(struct timeval* tv, struct timezone* tz)
{
	static LONGLONG birthunixhnsec = 116444736000000000;  /*in units of 100 ns */

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

bool compareArrays(Particle* d_particleArrayRes, Particle* particleArray)
{
	int i;
	for (i = 0; i < NUM_PARTICLES; i++)
	{
		if (d_particleArrayRes[i].velocity.x != particleArray[i].velocity.x ||
			d_particleArrayRes[i].velocity.y != particleArray[i].velocity.y ||
			d_particleArrayRes[i].velocity.z != particleArray[i].velocity.z ||
			d_particleArrayRes[i].position.x != particleArray[i].position.x ||
			d_particleArrayRes[i].position.y != particleArray[i].position.y ||
			d_particleArrayRes[i].position.z != particleArray[i].position.z)
			return false;
	}
	return true;
}

int main()
{
	printf("\nLet´s run exercise 3! \n\n");

	printf("\nNUM_PARTICLES:  %d ", NUM_PARTICLES);
	printf("\nNUM_ITERATIONS:  %d ", NUM_ITERATIONS);
	printf("\nThreads per block:  %d ", TPB);
	printf("\nNumber of thread blocks:  %d \n\n", (NUM_PARTICLES + TPB - 1) / TPB);

	Particle* d_particleArray;
	Particle* particleArray;

	cudaMalloc(&d_particleArray, NUM_PARTICLES * sizeof(Particle));

	if (d_particleArray == NULL)
	{
		printf("\n\nERROR 1! fail when allocating cuda dynamic memory!\n\n");
		return 1;
	}

	particleArray = (Particle*)malloc(NUM_PARTICLES * sizeof(Particle));

	if (particleArray == NULL)
	{
		printf("\n\nERROR 2! fail when allocating cuda dynamic memory!\n\n");
		return 1;
	}

	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		particleArray[i].position = make_float3(rand(), rand(), rand());
		particleArray[i].velocity = make_float3(rand(), rand(), rand());
	}

	cudaMemcpy(d_particleArray, particleArray, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

	// Let's measuare the time
	struct timeval tStart;
	struct timeval tEnd;

	printf("Computing GPU Particles Update...  \n");

	// We start the timer
	gettimeofday(&tStart, NULL);

	// Launch kernel to print hello worlds with Ids
	for(int i = 0; i <NUM_ITERATIONS; i++)
		kernelParticlesUpdate KERNEL_ARGS2((NUM_PARTICLES + TPB - 1) / TPB, TPB)(NUM_PARTICLES, d_particleArray);

	// We wait for the GPU
	cudaDeviceSynchronize();

	// We stop the timer...
	gettimeofday(&tEnd, NULL);

	// And finally print the timer
	printf("GPU Particles Update completed in %3.10f miliseconds \n", ((tEnd.tv_sec - tStart.tv_sec) * 1000000.0 + (tEnd.tv_usec - tStart.tv_usec)) / 1000.0);
	Particle* d_particleArrayRes = (Particle*)malloc(NUM_PARTICLES * sizeof(Particle));
	if (d_particleArrayRes == NULL)	{printf("\n\nERROR 2! fail when allocating cuda dynamic memory!\n\n");		return 1;}
	cudaMemcpy(d_particleArrayRes, d_particleArray, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);



	// Now let's go with the CPU Saxpy
	printf("\nComputing CPU Particles Update... \n");

	// We restart the timer 
	gettimeofday(&tStart, NULL);

	// The CPU Saxpy is computed
	for (int i = 0; i < NUM_ITERATIONS; i++)
	{
		cpuParticlesUpdate(particleArray);
	}

	// And then we stop the timer again
	gettimeofday(&tEnd, NULL);

	// Finally, we print the difference between the start time and the finished time
	printf("CPU Particles Update completed in %3.10f miliseconds \n", ((tEnd.tv_sec - tStart.tv_sec) * 1000000.0 + (tEnd.tv_usec - tStart.tv_usec)) / 1000.0);


	if (compareArrays(d_particleArrayRes, particleArray))
	{
		printf("\n\nBoth methods ARE equivalent since the resultant arrays HAVE the same data!\n\n");
	}
	else {
		printf("\n\nMethods ARE NOT equivalent because the resultant arrays HAVE NOT the same data!\n\n");
	}

	// Free resources of the 2 arrays that were dynamically allocated
	cudaFree(d_particleArray);
	free(particleArray);
	free(d_particleArrayRes);

	printf("\nExcercise 3 completed! \n");

	return 0;
}


