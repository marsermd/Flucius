#ifndef PSYSTEM_H
#define PSYSTEM_H
#include <glm\glm.hpp>
#include <thrust\device_vector.h>

#define SMOOTHING_CONST 1.56668147106f
#define CALC_KERNEL(r, h) (r > h ? 0 : powf(h * h - r * r, 3) * SMOOTHING_CONST / h * h * h * h)
#define GET_H(r) (r * 3)

#define PARTICLE_R 0.06f
#define PARTICLE_R2 0.0036f

struct Particle {
	glm::vec3 pos;
};

struct EmulatedParticles_Dev{
	thrust::device_vector<glm::vec3> positions;
	thrust::device_vector<glm::vec3> predPositions;
	thrust::device_vector<glm::vec3> velocities;
	thrust::device_vector<float> lambda;
	thrust::device_vector<glm::vec3> deltaPos;
	thrust::device_vector<glm::vec3> externalForces;
};

class PSystem {
public:
	PSystem();
	~PSystem();

	float calcKernel(float r) {
		float h = PARTICLE_R * 3;
		return CALC_KERNEL(r, h);
	}
	void addParticle(glm::vec3 pos);
	glm::vec3 * getParticles_dev();
	int getParticlesCount();

	void update();

private:
	EmulatedParticles_Dev * particles;
	void init();
};

#endif