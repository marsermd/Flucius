#ifndef PSYSTEM_H
#define PSYSTEM_H
#include <glm\glm.hpp>
#include <thrust\device_vector.h>

#include "Box.h"
#include "Partition3D.h"
#include "Renderable.h"

#include "PSystemStructures.h"

/*
 * PSystem is responsible for simulation of soft particle hydrodynamics
 */
class PSystem: public Renderable {
public:
	SPH::Settings settings;

	/*
	 * size -- size of side of the cube which encloses the simulation
	 */
	PSystem(float size);
	~PSystem();

	/*
	 * Add particle at pos
	 */
	void addParticle(glm::vec3 pos);

	/*
	 * Add a grid of particles filling a cubic box starting from startPos and placing linearCnt of particles on each side
	 */
	void addParticleBox(glm::vec3 startPos, int linearCnt);

	/*
	 * Get particles that are simulated
	 */
	SPH::Particle* getParticles_dev();

	/*
	 * Get particles positions
	 */
	glm::vec3* getParticlesPositions_dev();

	/*
	 * Get amount of simulated particles
	 */
	size_t getParticlesCount();

	/*
	 * Set renderer delegate that will do all the rendering for PSystem
	 */
	void setRenderer(Renderable* renderDelegate);

	virtual void render();

	/*
	 * Get enclosing box
	 */
	Box getBox();

private:
	Renderable* renderDelegate;
	Box box;
	SPH::EmulatedParticles_Thrust* particles_t;
	SPH::EmulatedParticles_Dev* particles_dev;
	Partition3D<SPH::Particle>* partition3D;

	void update();

	void cudaInit();
	void cudaClear();
};

#endif