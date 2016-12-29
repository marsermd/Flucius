#ifndef PARTITION3D_H
#define PARTITION3D_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "vectorHelper.h"

#include <math.h>
#include <GL/glew.h>
#include "Box.h"

#include "GridStructures.h"

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

	Partition3D<T>::Partition3D(T * elements_dev, int eCount, Box boundingBox, float radius) {
		r = radius;
		countx = ceil((boundingBox.size.x + EPS) / radius);
		county = ceil((boundingBox.size.y + EPS) / radius);
		countz = ceil((boundingBox.size.z + EPS) / radius);
		ttlCount = countx * county * countz;
		printf("%d, %d, %d\n", countx, county, countz);

		cudaInit(eCount);
		oldECount = eCount;
		update(elements_dev, eCount);
	}
	~Partition3D() {
		cudaCleanup();
	}

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