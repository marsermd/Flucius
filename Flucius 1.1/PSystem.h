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
	PSystem(float size);
	~PSystem();

	void addParticle(glm::vec3 pos);
	Particle* getParticles_dev();
	glm::vec3* getParticlesPositions_dev();
	int getParticlesCount();

	void setRenderer(Renderable* renderDelegate)
	{
		this->renderDelegate = renderDelegate;
	}

	virtual void render()
	{
		update();
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
	EmulatedParticles_Thrust * particles_t;
	EmulatedParticles_Dev * particles_dev;
	Partition3D<Particle> * partition3D;
	void init();
};

#endif