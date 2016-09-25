#include "neighbours.h"

#include "device_launch_parameters.h" 
#include "cudaHelper.h"

__global__ void findKNearestNeighbors(Particle* particles, int pCount, int * partitionIdx, int * partitions, dim3 counts, int ttlCount, int* neighbors, int* neighboursCnt, Box box){
	int index = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if(index < pCount) {
		int curNeighboursCnt = 0;
		glm::vec3 p = particles[index].pos;

		int x,y,z;
		x = GET_CLOSEST_POS(p.x, PARTICLE_H);
		y = GET_CLOSEST_POS(p.y, PARTICLE_H);
		z = GET_CLOSEST_POS(p.z, PARTICLE_H);


		float dist;
		float max;
		int m, idMax, beginID, curID, idx;
		glm::vec3 neighbourPos;

		for (int i = z - 1; i <= z + 1; i++) {
			for (int j = y - 1; j <= y + 1; j++) {
				for (int k = x - 1; k <= x + 1; k++) {
					idx = counts.z * counts.y * k + counts.z * j + i;
					if (idx < 0 || idx >= ttlCount) continue;

					beginID = partitions[idx];
					if (beginID < 0) continue;

					curID = beginID;
					while (curID < pCount && partitionIdx[beginID] == partitionIdx[curID]) {
						if (curID == index) {
							++curID;
							continue;
						}
						neighbourPos = particles[curID].pos;
						dist = glm::length(p - neighbourPos);

						if (curNeighboursCnt < MAX_NEIGHBOURS) {
							if (dist < PARTICLE_H) {
								neighbors[index * MAX_NEIGHBOURS + curNeighboursCnt] = curID;
								curNeighboursCnt++;
							}
						} else {
							max = glm::length(p - particles[neighbors[index * MAX_NEIGHBOURS]].pos);
							idMax = 0;
							for (m = 1; m < curNeighboursCnt; m++) {
								float d = glm::length(p - particles[neighbors[index * MAX_NEIGHBOURS + m]].pos); 
								if (d > max) {
									max = d;
									idMax = m;
								}
							}

							if (dist < max && dist < PARTICLE_H) {
								neighbors[index * MAX_NEIGHBOURS + idMax] = curID;
							}
						}

						++curID;
					}
				}
			}
		}
		neighboursCnt[index] = curNeighboursCnt;
	}
}

void cudaFindKNeighbors(Particle* particles_dev, int pCount, Partition3D<Particle> * partition3d, int * neighbours_dev, int * neighboursCnt_dev, Box box) {
	partition3d->update(particles_dev, pCount);
	dim3 counts(partition3d->countx, partition3d->county, partition3d->countz);
	findKNearestNeighbors<<<getBlocks(pCount), getThreads(pCount)>>>(particles_dev, pCount,
		partition3d->partitionIdx_dev, partition3d->partitions_dev, counts, partition3d->ttlCount,
		neighbours_dev, neighboursCnt_dev, box);

	checkCudaErrorsWithLine("failed finding nearest neighbours");
}