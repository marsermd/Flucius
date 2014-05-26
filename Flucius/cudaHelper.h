#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <glm\glm.hpp>
#include <cuda_runtime.h>

#define NEIGHBOURS_3D 27
#define THREADS_CNT 128

int getThreads (int runCount);

dim3 getBlocks (int runCount);

void checkCudaErrors(cudaError_t cudaStatus);

void checkCudaErrorsWithLine(char *message);

template <typename T>
T cudaGetEl(T* array, int id)
{
	T element;
	checkCudaErrors(cudaMemcpy((void *) &element, (void *)(array + id), sizeof(T), cudaMemcpyDeviceToHost));
	return element;
}

template <typename T>
T* cudaGetArr(T* array_dev, int size)
{
	T* arr = 0;
	cudaMallocHost((void **) &arr, size * sizeof(T));
	checkCudaErrors(cudaMemcpy((void *) arr, (void *)(array_dev), size * sizeof(T), cudaMemcpyDeviceToHost));
	return arr;
}
#endif