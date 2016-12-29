#ifndef PARTITION3D_H
#define PARTITION3D_H

#include "Box.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#define GET_CLOSEST_POS(coordinate, r) ((int)((coordinate + EPS) / (r)))

// T should have .pos attribute!
template <typename T> class Partition3D {
public:
	int countx, county, countz, ttlCount;
	float r;

	int* partitions_dev;
	int* partitionIdx_dev;
	int maxItemsPerPartition;

	Partition3D<T>::Partition3D(T * elements_dev, int eCount, Box boundingBox, float radius);
	~Partition3D();

	void update(T * elements_dev, int eCount);

	CUDA_CALLABLE_MEMBER
	bool contains (int x, int y, int z) {
		return x >= 0 && x < countx &&
			   y >= 0 && y < county &&
			   z >= 0 && z < countz;
	}

private:
	int oldECount;
	void cudaInit(int eCount);
	void cudaCleanup();

	void cudaCreateMemory(int eCount);
	void cudaFreeMemory();
};
#endif