#include "PSystem.h"

#include <thrust\device_ptr.h>
#include <thrust\detail\raw_pointer_cast.h>

#include "cudaHelper.h"
#include "PSystemConstants.h"

#define START_SIZE 32368
#define GRAVITY glm::vec3(0, -9.8f, 0)

EmulatedParticles_Thrust::EmulatedParticles_Thrust()
	//:
	//positions(START_COUNT),
	//particles(START_COUNT),
	//externalForces(START_COUNT),
	//neighbours(START_COUNT * MAX_NEIGHBOURS),
	//neighboursCnt(START_COUNT)
{
	Particle added = {
		glm::vec3(0.0f),
		0,
		glm::vec3(0.0f),
		glm::vec3(0.0f),
		0
	}; 
	positions.resize(START_SIZE, glm::vec3(0.0f));
	particles.resize(START_SIZE, added);
	externalForces.resize(START_SIZE, GRAVITY);
	neighbours.resize(START_SIZE * MAX_NEIGHBOURS, 0.0f);
	neighboursCnt.resize(START_SIZE, 0.0f);
	count = 0;
}

void EmulatedParticles_Thrust::setupDev(EmulatedParticles_Dev * particles_dev) {
	particles_dev->count = count;
	particles_dev->prevPos        = cudaGetRawRef<glm::vec3>(&positions);
	particles_dev->particles      =  cudaGetRawRef<Particle>(&particles);
	particles_dev->externalForces = cudaGetRawRef<glm::vec3>(&externalForces);
	particles_dev->neighbours     =       cudaGetRawRef<int>(&neighbours);
	particles_dev->neighboursCnt  =       cudaGetRawRef<int>(&neighboursCnt);
}

PSystem::PSystem(float size) :
	box(glm::vec3(0), glm::vec3(size)),
	particles_t(new EmulatedParticles_Thrust()),
	particles_dev(new EmulatedParticles_Dev())
{

	/*addParticle(glm::vec3(5.5f, 5.5f, 5.5f));
	addParticle(glm::vec3(5.5f, 5.5f, 0.5f));
	addParticle(glm::vec3(5.5f, 0.5f, 5.5f));
	addParticle(glm::vec3(5.5f, 0.5f, 0.5f));
	addParticle(glm::vec3(0.4f, 5.5f, 5.5f));
	addParticle(glm::vec3(0.4f, 5.5f, 0.5f));
	addParticle(glm::vec3(0.5f, 0.5f, 5.5f));
	addParticle(glm::vec3(0.5f, 0.5f, 0.5f));*/
	//addParticle(glm::vec3(1 + 1 * PARTICLE_R * 2.0f, 1, 1 + 1 * PARTICLE_R * 2.0f));
	//addParticle(glm::vec3(1 + 1 * PARTICLE_R * 1, 1, 1 + 1 * PARTICLE_R * 1));
	//addParticle(glm::vec3(5 * PARTICLE_R + 1 * PARTICLE_R, 5 * PARTICLE_R, 5 * PARTICLE_R + 3 * PARTICLE_R));

	const int CNT = 20;
	float dist = PARTICLE_H * 1.0f;
	for (int x = 2; x < CNT; x++) {
		for (int y = 10; y < 10 + CNT; y++) {
			for (int z = 2; z < CNT; z++) {
				addParticle(glm::vec3(x * dist + rand() % 100 * 0.01f , 2.0f + y * dist + rand() % 100 * 0.01f, z * dist + rand() % 100 * 0.01f));
				//addParticle(glm::vec3(1.0f + x * PARTICLE_R, 2.0f + x * (z - 18) * (z - 17) * PARTICLE_R / 300, 1.0f + z * PARTICLE_R));
			}
		}
	}

	particles_t->setupDev(particles_dev);

	partition3D = new Partition3D<Particle>(particles_dev->particles, particles_dev->count, box, PARTICLE_H);
}

PSystem::~PSystem() {
	delete particles_dev;
	delete particles_t;
}

glm::vec3 * PSystem::getParticles_dev() {
	return thrust::raw_pointer_cast(particles_t->positions.data());
}

void PSystem::addParticle(glm::vec3 pos) {
	Particle added = {
		pos,
		particles_t->count,
		glm::vec3(0.0f),
		glm::vec3(0.0f),
		0
	};
	if (particles_t->count == particles_t->positions.size()) {
		int newSize = max(particles_t->count * 2, 100);
		particles_t->positions.resize(newSize, glm::vec3(0.0f));
		particles_t->particles.resize(newSize, added);
		particles_t->externalForces.resize(newSize, GRAVITY);
		particles_t->neighbours.resize(newSize * MAX_NEIGHBOURS, 0);
		particles_t->neighboursCnt.resize(newSize, 0);
	}
	particles_t->positions[particles_t->count] = glm::vec3(pos);
	particles_t->particles[particles_t->count] = added;

	particles_t->count++;
	//checkCudaErrorsWithLine("adding particles failed!");

}

int PSystem::getParticlesCount() {
	return particles_t->count;
}