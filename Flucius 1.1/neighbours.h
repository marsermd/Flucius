#ifndef NEIGHBOURS_H
#define NEIGHBOURS_H

#include "Partition3D.h"
#include "PSystemStructures.h"


/**
 *	particles_dev -- particles to find neighbours for
 *  pCount -- count of particles
 *  partition3d -- partition 3d for particles_dev
 *  neighbours_dev -- result of cudaFindKNeighbors. array of length pCount * MAX_NEIGHBOURS. 
 *		At place i * MAX_NEIGHBOURS to i * MAX_NEIGHBOURS + MAX_NEIGHBOURS - 1
 *		are stored id's of neighbours for i-th particle
 *	neighboursCnt_dev -- result of cudaFindKNeighbors. neighboursCnt_dev[i] is the count of neighbours found for i-th particle
 */
void cudaFindKNeighbors(SPH::Particle* particles_dev, int pCount, Partition3D<SPH::Particle> * partition3d, int * neighbours_dev, int * neighboursCnt_dev);

#endif