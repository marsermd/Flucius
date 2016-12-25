#ifndef PSYSTEMSTRUCTURES_H
#define PSYSTEMSTRUCTURES_H

#include <glm\glm.hpp>
#include <thrust\device_vector.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

struct Particle{
	int id;
	glm::vec3 pos;
	glm::vec3 deltaPos;
	glm::vec3 velocity;
	glm::vec3 nextVelocity;
	float lambda; 

	CUDA_CALLABLE_MEMBER Particle()
	{
		id = 0;
		pos = glm::vec3(0);
		deltaPos = glm::vec3(0);
		velocity = glm::vec3(0);
		nextVelocity = glm::vec3(0);
		lambda = 0;
	}
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