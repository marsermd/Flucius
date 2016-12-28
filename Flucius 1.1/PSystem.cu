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
	count = 0;
}

void EmulatedParticles_Thrust::pushToDevice()
{
	particles = thrust::device_vector<Particle>(particles_host);
	prev_pos = thrust::device_vector<glm::vec3>(positions_host);

	vorticities.resize(count, glm::vec3(0.0f));
	externalForces.resize(count, GRAVITY);
	neighbours.resize(count * MAX_NEIGHBOURS, 0);
	neighboursCnt.resize(count, 0);
}

void EmulatedParticles_Thrust::setupDev(EmulatedParticles_Dev * particles_dev) 
{
	particles_dev->count = count;

	particles_dev->prevPos        = cudaGetRawRef<glm::vec3>(&prev_pos);
	particles_dev->vorticities    = cudaGetRawRef<glm::vec3>(&vorticities);
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
	int cnt = 30;
	float dist = PARTICLE_H * 0.6f;
	float randomScale = 0.01f;
	for (int x = 0; x < cnt; x++) {
		for (int y = 0; y < cnt; y++) {
			for (int z = 0; z < cnt; z++) {
				glm::vec3 pos = glm::vec3(20, 0, 20) + glm::vec3(x, y, z) * dist + 
					glm::vec3(
						(rand() % 1000) * 0.001f * randomScale,
						(rand() % 1000) * 0.001f * randomScale,
						(rand() % 1000) * 0.001f * randomScale
					);
				addParticle(pos);
			}
		}
	}

	particles_t->pushToDevice();
	particles_t->setupDev(particles_dev);

	partition3D = new Partition3D<Particle>(particles_dev->particles, particles_dev->count, box, PARTICLE_H);
}

PSystem::~PSystem() {
	delete particles_dev;
	delete particles_t;
}

Particle* PSystem::getParticles_dev() {
	return particles_dev->particles;
}

glm::vec3* PSystem::getParticlesPositions_dev() {
	return particles_dev->prevPos;
}

void PSystem::addParticle(glm::vec3 pos) {
	Particle added;
	added.pos = pos;
	added.id = particles_t->count;

	particles_t->positions_host.push_back(pos);
	particles_t->particles_host.push_back(added);

	particles_t->count++;
}

int PSystem::getParticlesCount() {
	return particles_t->count;
}