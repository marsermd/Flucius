#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include <thrust\device_ptr.h>
#include <thrust\detail\raw_pointer_cast.h>

#include "PSystemStructures.h"

#define NEIGHBOURS_3D 27
#define THREADS_CNT 128

int getThreads(int runCount, int threads_cnt = THREADS_CNT);

dim3 getBlocks(int runCount, int threads_cnt = THREADS_CNT);

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

template <typename T>
T* cudaGetRawRef(thrust::device_vector<T> * vector)
{
	return thrust::raw_pointer_cast(vector->data());
}

#endif