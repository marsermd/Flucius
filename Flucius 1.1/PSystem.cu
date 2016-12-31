#include "PSystem.h"

#include <thrust\copy.h>

#include "cudaHelper.h"
#include "PSystemConstants.h"

using namespace SPH;

EmulatedParticles_Thrust::EmulatedParticles_Thrust()
{
	count = 0;
}

void EmulatedParticles_Thrust::pushToDevice()
{
	size_t curDeviceCnt = particles.size();
	if (curDeviceCnt == count)
	{
		return;
	}

	particles.resize(count, Particle());
	thrust::copy(particles_host.begin(), particles_host.end(), particles.begin() + curDeviceCnt);
	particles_host.clear();

	prevPos.resize(count, glm::vec3(0));
	thrust::copy(prevPos_host.begin(), prevPos_host.end(), prevPos.begin() + curDeviceCnt);
	prevPos_host.clear();

	vorticities.resize(count, glm::vec3(0.0f));
	neighbours.resize(count * MAX_NEIGHBOURS, 0);
	neighboursCnt.resize(count, 0);
}

void EmulatedParticles_Thrust::setupDev(EmulatedParticles_Dev * particles_dev) 
{
	pushToDevice();
	particles_dev->count = count;

	particles_dev->prevPos        = cudaGetRawRef<glm::vec3>(&prevPos);
	particles_dev->vorticities    = cudaGetRawRef<glm::vec3>(&vorticities);
	particles_dev->particles      =  cudaGetRawRef<Particle>(&particles);
	particles_dev->neighbours     =       cudaGetRawRef<int>(&neighbours);
	particles_dev->neighboursCnt  =       cudaGetRawRef<int>(&neighboursCnt);
}

PSystem::PSystem(float size) :
	box(glm::vec3(0), glm::vec3(size)),
	particles_t(new EmulatedParticles_Thrust()),
	particles_dev(new EmulatedParticles_Dev())
{
	addParticleBox(glm::vec3(10, 5, 10), 30);
	addParticleBox(glm::vec3(40, 0, 40), 10, true);

	settings.deltaQ = 0.1f * PARTICLE_H;
	settings.iterationsCount = 3;
	settings.pressureK = 0.01f;
	settings.setRestDencity(0.5f);
	settings.relaxation = 5.0f;
	settings.viscosity = 0.01f;
	settings.vorticityEpsilon = 3.0f;
	settings.setGravity(glm::vec3(0, -10, 0));

	particles_t->setupDev(particles_dev);

	partition3D = new Partition3D<Particle>(particles_dev->particles, particles_dev->count, box, PARTICLE_H);

	cudaInit();
}

PSystem::~PSystem() {
	cudaClear();

	delete particles_dev;
	delete particles_t;
}

void PSystem::addParticleBox(glm::vec3 startPos, int linearCnt, bool isFixed)
{
	float dist = PARTICLE_H * 0.6f;
	float randomScale = 0.01f;
	for (int x = 0; x < linearCnt; x++) {
		for (int y = 0; y < linearCnt; y++) {
			for (int z = 0; z < linearCnt; z++) {
				glm::vec3 pos = startPos + glm::vec3(x, y, z) * dist +
					glm::vec3(
					(rand() % 1000) * 0.001f * randomScale,
					(rand() % 1000) * 0.001f * randomScale,
					(rand() % 1000) * 0.001f * randomScale
					);
				addParticle(pos, isFixed);
			}
		}
	}
}


Particle* PSystem::getParticles_dev() {
	return particles_dev->particles;
}

glm::vec3* PSystem::getParticlesPositions_dev() {
	return particles_dev->prevPos;
}

void PSystem::addParticle(glm::vec3 pos, bool isFixed) {
	Particle added(isFixed);
	added.pos = pos;
	added.id = particles_t->count;

	particles_t->prevPos_host.push_back(pos);
	particles_t->particles_host.push_back(added);

	particles_t->count++;
}

size_t PSystem::getParticlesCount() {
	return particles_t->count;
}

void PSystem::setRenderer(Renderable* renderDelegate)
{
	this->renderDelegate = renderDelegate;
}

void PSystem::render()
{
	update();
	renderDelegate->modelMatrixID = modelMatrixID;
	renderDelegate->render();
}

Box PSystem::getBox()
{
	return box;
}
