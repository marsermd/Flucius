#include "PSystem.h"

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\device_ptr.h>
#include <thrust\detail\raw_pointer_cast.h>

#define EPS 1.0e-6f

thrust::device_vector<Particle> particles_dev;
thrust::host_vector<Particle> particles; 

void Particle::update()  {
	glm::mat4 rotate = glm::translate(glm::rotate(glm::translate(3.0f, 3.0f, 3.0f), (pos.x - 3) * (pos.x - 3) + (pos.y - 3) * (pos.y - 3), 1.0f, 0.0f, 0.0f), glm::vec3(-3.0f, -3.0f, -3.0f));
	pos = glm::vec3(rotate * glm::vec4(pos, 1));
}


void PSystem::init() {
	//addParticle(glm::vec3(5.5f, 5.5f, 5.5f));
	//addParticle(glm::vec3(5.5f, 5.5f, 0.5f));
	//addParticle(glm::vec3(5.5f, 0.5f, 5.5f));
	//addParticle(glm::vec3(5.5f, 0.5f, 0.5f));
	//addParticle(glm::vec3(0.4f, 5.5f, 5.5f));
	//addParticle(glm::vec3(0.4f, 5.5f, 0.5f));
	//addParticle(glm::vec3(0.5f, 0.5f, 5.5f));
	//addParticle(glm::vec3(0.5f, 0.5f, 0.5f));

	for (int x = 4; x < 45; x++) {
		for (int y = 4; y < 32; y++) {
			for (int z = 4; z < 32; z++) {
				addParticle(glm::vec3(1 + x * PARTICLE_R, 1 + y * PARTICLE_R, 1 + z * PARTICLE_R));
			}
		}
	}
}

PSystem::~PSystem() {
}

Particle * PSystem::getParticles_dev() {
	return thrust::raw_pointer_cast(particles_dev.data());
}

Particle PSystem::addParticle(glm::vec3 pos) {
	Particle p = {glm::vec3(pos), defaultR};
	particles.push_back(p);
	return p;
}

int PSystem::getParticlesCount() {
	return particles_dev.size();
}

void PSystem::update() {
		for (int i = 0; i < particles.size(); i++) {
			particles[i].update();
		}
		particles_dev = particles;
}