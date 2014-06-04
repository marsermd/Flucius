#include "PSystem.h"

#include <thrust\device_ptr.h>
#include <thrust\detail\raw_pointer_cast.h>

#define EPS 1.0e-6f

PSystem::PSystem() {
	//addParticle(glm::vec3(5.5f, 5.5f, 5.5f));
	//addParticle(glm::vec3(5.5f, 5.5f, 0.5f));
	//addParticle(glm::vec3(5.5f, 0.5f, 5.5f));
	//addParticle(glm::vec3(5.5f, 0.5f, 0.5f));
	//addParticle(glm::vec3(0.4f, 5.5f, 5.5f));
	//addParticle(glm::vec3(0.4f, 5.5f, 0.5f));
	//addParticle(glm::vec3(0.5f, 0.5f, 5.5f));
	//addParticle(glm::vec3(0.5f, 0.5f, 0.5f));
	particles = new EmulatedParticles_Dev();

	for (int x = 4; x < 64; x++) {
		for (int y = 4; y < 32; y++) {
			for (int z = 4; z < 32; z++) {
				addParticle(glm::vec3(1 + x * PARTICLE_R, 1 + y * PARTICLE_R, 1 + z * PARTICLE_R));
			}
		}
	}
}

PSystem::~PSystem() {
}

glm::vec3 * PSystem::getParticles_dev() {
	return thrust::raw_pointer_cast(particles->positions.data());
}

void PSystem::addParticle(glm::vec3 pos) {
	particles->positions.push_back(glm::vec3(pos));
	particles->predPositions.push_back(glm::vec3(pos));
	particles->velocities.push_back(glm::vec3(0));
	particles->lambda.push_back(0);
	particles->deltaPos.push_back(glm::vec3(0));
	particles->externalForces.push_back(glm::vec3(0));
}

int PSystem::getParticlesCount() {
	return particles->positions.size();
}