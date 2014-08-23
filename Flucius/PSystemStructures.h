#ifndef PSYSTEMSTRUCTURES_H
#define PSYSTEMSTRUCTURES_H

#include <glm\glm.hpp>
#include <thrust\device_vector.h>

#define PI 3.1415f
#define POLY6_CONST 1.5666814f
#define SPIKY_CONST 14.323944f

#define PARTICLE_H 0.18f
#define PARTICLE_H6 (float)(PARTICLE_H * PARTICLE_H * PARTICLE_H * PARTICLE_H * PARTICLE_H * PARTICLE_H)
#define PARTICLE_H9 (float)(PARTICLE_H6 * PARTICLE_H * PARTICLE_H * PARTICLE_H)

#define PARTICLE_R 0.06f

#define CALC_KERNEL(r) (r > PARTICLE_H ? 0 : powf(PARTICLE_H * PARTICLE_H - r * r, 3) * POLY6_CONST / (PARTICLE_H * PARTICLE_H * PARTICLE_H * PARTICLE_H))

#define MAX_NEIGHBOURS 30
#define RELAXATION .01f // epselon, or relaxation parameter

struct Particle{
	glm::vec3 pos;
	int id;
	glm::vec3 velocity;
	glm::vec3 deltaPos;
	float lambda; 
};

struct EmulatedParticles_Dev{
	int count;
	glm::vec3 * prevPos;
	Particle * particles;
	glm::vec3 * externalForces; 
	int* neighbours;
	int* neighboursCnt;
};

struct EmulatedParticles_Thrust{
	int count;
	thrust::device_vector<glm::vec3> positions;
	thrust::device_vector<Particle> particles;
	thrust::device_vector<glm::vec3> externalForces;
	thrust::device_vector<int> neighbours;
	thrust::device_vector<int> neighboursCnt;

	EmulatedParticles_Thrust();
	void setupDev(EmulatedParticles_Dev * particles);
};

#endif