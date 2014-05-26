#include "cudaHelper.h"

int getThreads (int runCount) {
	return runCount > THREADS_CNT ? THREADS_CNT : runCount;
}

dim3 getBlocks (int runCount) {
	runCount = (runCount - 1) / THREADS_CNT + 1;
	int blocksx = 1;
	int blocksy = 1;
	if (runCount > 65535) {
		blocksx = 65535;
		blocksy = (runCount - 1) / 65535 + 1;
	} else {
		blocksx = runCount;
	}
	return dim3(blocksx, blocksy);
}

void checkCudaErrors(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda failed %s\n", cudaGetErrorString(cudaStatus));
	}
}

void checkCudaErrorsWithLine(char *message) {
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda failed. Message:\n%s\n erroro:\n%s", message, cudaGetErrorString(cudaStatus));
	}
}