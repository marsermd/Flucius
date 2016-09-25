#include "Partition3D.h"
#ifdef PARTITION3D_H

#include <cuda_runtime.h>
#include "device_launch_parameters.h" 
#include <thrust\device_ptr.h>
#include <thrust\detail\raw_pointer_cast.h>
#include <thrust\sort.h>

#include "cudaHelper.h"

__global__ void clearPartitions(int* partitons, int ttlCount)
{
	int index = min(blockIdx.y * 65535 * THREADS_CNT + blockIdx.x * THREADS_CNT + threadIdx.x, ttlCount - 1);
	partitons[index] = -1;
}

// Finds for Elements the partition index for the cell in which the element is placed
template <typename T>
__global__ void findElementPartitionIndex(T * elements, int * partitonIdx, int eCount, float r, dim3 counts)
{
	int index = min(blockIdx.y * 65535 * THREADS_CNT + blockIdx.x * THREADS_CNT + threadIdx.x, eCount - 1);

	int x, y, z;
	glm::vec3 p = elements[index].pos;
	x = GET_CLOSEST_POS(p.x, r);
	y = GET_CLOSEST_POS(p.y, r);
	z = GET_CLOSEST_POS(p.z, r);
	partitonIdx[index] = counts.z * counts.y * x + counts.z * y + z;
}


// updating partition values(each partition pointers on the first element)
__global__ void matchElementToCell(int * partitonIdx, int * partitons, int eCount, int ttlCount)
{
	int index = min(blockIdx.y * 65535 * THREADS_CNT + blockIdx.x * THREADS_CNT + threadIdx.x, eCount - 1);

	if (index == 0) {
		partitons[partitonIdx[index]] = index;
	} else if(partitonIdx[index] != partitonIdx[index - 1] &&
		partitonIdx[index] >= 0 && partitonIdx[index] < ttlCount) {
			partitons[partitonIdx[index]] = index;
	}
}
//________________________________________HOST CODE___________________________
template <typename T>
void Partition3D<T>::update(T * elements_dev, int eCount) {
	clearPartitions<<<getBlocks(ttlCount), getThreads(ttlCount)>>>(partitions_dev, ttlCount);
	checkCudaErrorsWithLine("failed clearing partition");

	dim3 counts(countx, county, countz);

	findElementPartitionIndex<T><<<getBlocks(eCount), getThreads(eCount)>>>(elements_dev, partitionIdx_dev, eCount, r, counts);
	checkCudaErrorsWithLine("failed calculating elements positions");

	thrust::device_ptr<int> partitionIdx_t = thrust::device_pointer_cast<int>(partitionIdx_dev);
	thrust::device_ptr<T> elements_t = thrust::device_pointer_cast<T>(elements_dev);
	thrust::sort_by_key(partitionIdx_t, partitionIdx_t + eCount, elements_t);
	checkCudaErrorsWithLine("failed thrust sorting");

	matchElementToCell<<<getBlocks(eCount), getThreads(eCount)>>>(partitionIdx_dev, partitions_dev, eCount, ttlCount);
	checkCudaErrorsWithLine("failed matching elements");

	thrust::device_vector<int> counts_t(eCount);
	thrust::uninitialized_fill(counts_t.begin(), counts_t.end(), 1);
	checkCudaErrorsWithLine("failed setting up counts");
	thrust::exclusive_scan_by_key(partitionIdx_t, partitionIdx_t + eCount, counts_t.begin(), counts_t.begin());
	checkCudaErrorsWithLine("failed scaning matched elements");
	maxItemsPerPartition = thrust::reduce(counts_t.begin(), counts_t.end(), (int)0, thrust::maximum<int>()) + 1;
	checkCudaErrorsWithLine("failed counting maxItemsPerPartition");
}

template <typename T>
void Partition3D<T>::cudaInit(int eCount) {
	cudaCreateMemory(eCount);
}

template <typename T>
void Partition3D<T>::cudaCleanup() {
	cudaFreeMemory();
}

template <typename T>
void Partition3D<T>::cudaCreateMemory(int eCount) {
	cudaMalloc((void **) &partitions_dev, ttlCount*sizeof(int));
	cudaMalloc((void **) &partitionIdx_dev, eCount*sizeof(int));
}

template <typename T>
void Partition3D<T>::cudaFreeMemory() {
	cudaFree(partitions_dev);
	cudaFree(partitionIdx_dev);
}


template <typename T> 
void createTEMPLATE(){
	T elements[2];
	T * elements_dev; 
	cudaMalloc((void **) &elements_dev, sizeof(T) * 2);
	cudaMemcpy((void *)elements_dev, (void *)elements, sizeof(T) * 2, cudaMemcpyDeviceToHost);
	Partition3D<T> * p3d = new Partition3D<T>(elements_dev, 2, Box(glm::vec3(-1), 2), 0.25f);
	p3d->update(elements_dev, 2);
	p3d->contains(0, 0, 0);
	delete p3d;
}


#include "GridStructures.h"
#include "PSystemStructures.h"
// This function is used ONLY to make compiler generate template of given type.
// NO, REALLY, DON'T CALL IT! I MEAN IT! AND GOD SAVE YOU IF YOU USE IT
void dontCALLthisFUNCTION_IT_IsUsLeSS() {
	//register here any type you want to use
	createTEMPLATE<GridVertex>();
	createTEMPLATE<Particle>();
}

#endif