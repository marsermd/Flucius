#ifndef NEIGHBOURS_H
#define NEIGHBOURS_H

#include "Partition3D.h"
#include "PSystemStructures.h"

void cudaFindKNeighbors(Particle* particles_dev, int pCount, Partition3D<Particle> * partition3d, int * neighbours_dev, int * neighboursCnt_dev);

#endif