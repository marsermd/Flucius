#ifndef PSYSTEM_H
#define PSYSTEM_H
#include <glm\glm.hpp>
#include <glm\gtx\transform.hpp>

#define SMOOTHING_CONST 1.56668147106f
#define CALC_KERNEL(r, h) (r > h ? 0 : powf(h * h - r * r, 3) * SMOOTHING_CONST / h * h * h * h)
#define GET_H(r) (r * 3)

#define PARTICLE_R 0.04f
#define PARTICLE_R2 0.0016f

struct Particle {
	glm::vec3 pos;
	float r;
	float getH() {
		return GET_H(r);
	}
	void update();
};

class PSystem {
public:
	PSystem() : defaultR(PARTICLE_R){
		init();
	}
	~PSystem();

	static float calcKernel(float r, float h) {
		return CALC_KERNEL(r, h);
	}

	float calcKernel(float r) {
		float h = defaultR * 3;
		return calcKernel(r, h);
	}
	Particle addParticle(glm::vec3 pos);
	Particle * getParticles_dev();
	int getParticlesCount();

	float defaultR;

	void update();

private:
	void init();
};

#endif