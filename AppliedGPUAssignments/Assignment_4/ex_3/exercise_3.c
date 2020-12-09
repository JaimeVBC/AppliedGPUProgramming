// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>
#include <Windows.h>

// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

#define ARRAY_SIZE 10000
#define NUM_ITERATIONS 100000
// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

//Particle structure
// Device declaration
/*struct Particle
{
	cl_float3 pos;
	cl_float3 vel;
};*/

// Host declaration
typedef struct
{
	cl_float3 pos;
	cl_float3 vel;
}Particle;


const char* mykernel =
"struct Particle			\n"
"{							\n"
"	float3 pos;			\n"
"	float3 vel;			\n"
"};							\n"
"__kernel					\n"
"void particles_gpu (__global struct Particle *X, int array_size_aux){				\n"
"int index = get_global_id(0);					\n"
"if(index < array_size_aux){					\n"
"X[index].vel.x += 0.5 * index;					\n"
"X[index].vel.y += 2 * index;					\n"
"X[index].vel.z += 0.75 * index;				\n"
"X[index].pos.x += X[index].vel.x;				\n"
"X[index].pos.y += X[index].vel.y; 				\n"
"X[index].pos.z += X[index].vel.z;			  }}\n";


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

int compare_particles(Particle* X, Particle* Y)
{
	int i;
	for (i = 0; i < ARRAY_SIZE; i++)
	{
		if (abs(X[i].pos.x - Y[i].pos.x) > 0.001f) return 0;
		if (abs(X[i].pos.y - Y[i].pos.y) > 0.001f) return 0;
		if (abs(X[i].pos.z - Y[i].pos.z) > 0.001f) return 0;
		if (abs(X[i].vel.x - Y[i].vel.x) > 0.001f) return 0;
		if (abs(X[i].vel.y - Y[i].vel.y) > 0.001f) return 0;
		if (abs(X[i].vel.z - Y[i].vel.z) > 0.001f) return 0;
	}

	return 1;

}

void particles_CPU(Particle* X_CPU)
{
	int i;
	for (i = 0; i < ARRAY_SIZE; i++)
	{
		X_CPU[i].vel.x += 0.5 * i;
		X_CPU[i].vel.y += 2 * i;
		X_CPU[i].vel.z += 0.75 * i;
		X_CPU[i].pos.x += X_CPU[i].vel.x;
		X_CPU[i].pos.y += X_CPU[i].vel.y;
		X_CPU[i].pos.z += X_CPU[i].vel.z;
	}
}

int main(int argc, char *argv) {

  struct timeval tStart;
  struct timeval tEnd;

  cl_platform_id * platforms; cl_uint     n_platform;

  // Find OpenCL Platforms
  cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
  err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

  // Find and sort devices
  cl_device_id *device_list; cl_uint n_devices;
  err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);CHK_ERROR(err);
  device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
  err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);CHK_ERROR(err);
  
  // Create and initialize an OpenCL context
  cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);CHK_ERROR(err);

  // Create a command queue
  if (device_list == NULL) return;
  cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);CHK_ERROR(err);

  //initialize arrays
  int array_size = ARRAY_SIZE * sizeof(Particle);

  Particle *X =	 (Particle*)malloc(array_size);
  if (X == NULL) return;
  Particle *X_CPU =	 (Particle*)malloc(array_size);
  if (X_CPU == NULL) return;

  int i;
  for (i = 0; i < ARRAY_SIZE; i++)
  {
	  X[i].pos.x = rand();
	  X[i].pos.y = rand();
	  X[i].pos.z = rand();

	  X[i].vel.x = rand();
	  X[i].vel.y = rand();
	  X[i].vel.z = rand();

	  X_CPU[i].pos.x = X[i].pos.x;
	  X_CPU[i].pos.y = X[i].pos.y;
	  X_CPU[i].pos.z = X[i].pos.z;

	  X_CPU[i].vel.x = X[i].vel.x;
	  X_CPU[i].vel.y = X[i].vel.y;
	  X_CPU[i].vel.z = X[i].vel.z;

  }

//Create buffers in the GPU
  cl_mem X_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size, NULL, &err);

//Enqueue the array in the GPU
  err = clEnqueueWriteBuffer(cmd_queue, X_dev, CL_TRUE, 0, array_size, X, 0, NULL, NULL); CHK_ERROR(err);

// create program
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&mykernel, NULL, &err);

  err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
	  size_t len;
	  char buffer[2048];
	  clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
	  fprintf(stderr, "Build error: %s\n", buffer);
	  return 0;
  }

  cl_kernel kernel = clCreateKernel(program, "particles_gpu", &err);

  //Set the 3 arguments of our kernel
  int array_size_aux = ARRAY_SIZE;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&X_dev); CHK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(int), (void*)&array_size_aux); CHK_ERROR(err);

  size_t workgroup_size = 256;
  size_t num_blocks = (ARRAY_SIZE + workgroup_size - 1) / workgroup_size;
  size_t n_workitem = num_blocks * workgroup_size;

  // CPU implementation
  gettimeofday(&tStart, NULL);
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
	  particles_CPU(X_CPU);
  }
  gettimeofday(&tEnd, NULL);
  printf("CPU particles completed in %3.10f miliseconds \n", ((tEnd.tv_sec - tStart.tv_sec) * 1000000.0 + (tEnd.tv_usec - tStart.tv_usec)) / 1000.0);


  // Kernel launch
  gettimeofday(&tStart, NULL);
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
	  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &n_workitem, &workgroup_size, 0, NULL, NULL); CHK_ERROR(err);
  }
  
  //Transfer the data back to the host
  err = clEnqueueReadBuffer(cmd_queue, X_dev, CL_TRUE, 0, array_size, X, 0, NULL, NULL); CHK_ERROR(err);
  gettimeofday(&tEnd, NULL);
  
  printf("GPU particles completed in %3.10f miliseconds \n", ((tEnd.tv_sec - tStart.tv_sec) * 1000000.0 + (tEnd.tv_usec - tStart.tv_usec)) / 1000.0);



  // Arrays comparison
  if (compare_particles(X, X_CPU))
  {
	  printf("Arrays ARE equivalent");
  }
  else
  {
	  printf("Arrays ARE NOT equivalent");
  }

  //Finish
  err = clFlush(cmd_queue); CHK_ERROR(err);
  err = clFinish(cmd_queue); CHK_ERROR(err);

  // Finally, release all that we have allocated.
  err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err); CHK_ERROR(err);
  err = clReleaseContext(context);CHK_ERROR(err); CHK_ERROR(err);
  free(platforms);
  free(device_list);
  free(X);
  free(X_CPU);
  
  return 0;
}



// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
  switch (errorCode) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -71: return "CL_INVALID_SPEC_ID";
  case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
  case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
  case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
  case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
  case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
  case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
  case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
  case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
  case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
  case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
  case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
  case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
  case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
  case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
  case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
  case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
  case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
  case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
  default: return "CL_UNKNOWN_ERROR";
  }
}
