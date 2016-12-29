#ifndef PSYSTEM_H
#define PSYSTEM_H
#include <glm\glm.hpp>
#include <thrust\device_vector.h>

#include "Box.h"
#include "Partition3D.h"
#include "Renderable.h"

#include "PSystemStructures.h"

class PSystem: public Renderable {
public:
	SPH::Settings settings;

	PSystem(float size);
	~PSystem();

	void addParticle(glm::vec3 pos);
	void addParticleBox(glm::vec3 startPos, int linearCnt);
	SPH::Particle* getParticles_dev();
	glm::vec3* getParticlesPositions_dev();
	size_t getParticlesCount();

	void setRenderer(Renderable* renderDelegate)
	{
		this->renderDelegate = renderDelegate;
	}

	virtual void render()
	{
		update();
		renderDelegate->modelMatrixID = modelMatrixID;
		renderDelegate->render();
	}

	void update();

	Box getBox()
	{
		return box;
	}

private:
	Renderable* renderDelegate;
	Box box;
	SPH::EmulatedParticles_Thrust * particles_t;
	SPH::EmulatedParticles_Dev * particles_dev;
	Partition3D<SPH::Particle> * partition3D;

	void cudaInit();
	void cudaClear();
};

#endif