#ifndef PSYSTEM_H
#define PSYSTEM_H
#include <glm\glm.hpp>
#include <thrust\device_vector.h>

#include "Box.h"
#include "Partition3D.h"

#include "PSystemStructures.h"

class PSystem {
public:
	PSystem(float size);
	~PSystem();

	void addParticle(glm::vec3 pos);
	glm::vec3 * getParticles_dev();
	int getParticlesCount();

	void update();

private:
	Box box;
	EmulatedParticles_Thrust * particles_t;
	EmulatedParticles_Dev * particles_dev;
	Partition3D<Particle> * partition3D;
	void init();
};

#endif