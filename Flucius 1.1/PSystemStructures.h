#ifndef PSYSTEMSTRUCTURES_H
#define PSYSTEMSTRUCTURES_H

#include <glm\glm.hpp>
#include <thrust\device_vector.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

namespace SPH
{
	struct Settings
	{
	public:
		CUDA_CALLABLE_MEMBER Settings(){}
		CUDA_CALLABLE_MEMBER ~Settings(){}

		void setRestDencity(float value);
		void setGravity(glm::vec3 value);
		CUDA_CALLABLE_MEMBER glm::vec3 getGravity()
		{
			return glm::vec3(gravity[0], gravity[1], gravity[2]);
		}

		int iterationsCount;
		float deltaQ;
		float pressureK;
		float oneOverRestDencity;
		float relaxation;
		float viscosity;
		float vorticityEpsilon;
		float gravity[3];
	};
}

namespace SPH
{
	struct Particle
	{
		bool isFixed;
		int id;
		glm::vec3 pos;
		glm::vec3 deltaPos;
		glm::vec3 velocity;
		glm::vec3 nextVelocity;
		float lambda;

		CUDA_CALLABLE_MEMBER Particle(bool isFixed = false):
			isFixed(isFixed)
		{

			id = 0;
			pos = glm::vec3(0);
			deltaPos = glm::vec3(0);
			velocity = glm::vec3(0);
			nextVelocity = glm::vec3(0);
			lambda = 0;
		}
	};
}

namespace SPH
{
	struct EmulatedParticles_Dev
	{
		int count;
		glm::vec3* prevPos;
		glm::vec3* vorticities;
		Particle* particles;
		int* neighbours;
		int* neighboursCnt;
	};
}

namespace SPH
{
	struct EmulatedParticles_Thrust
	{
		size_t count;
		thrust::host_vector<glm::vec3> prevPos_host;
		thrust::host_vector<Particle> particles_host;

		thrust::device_vector<glm::vec3> prevPos;
		thrust::device_vector<glm::vec3> vorticities;
		thrust::device_vector<Particle> particles;
		thrust::device_vector<int> neighbours;
		thrust::device_vector<int> neighboursCnt;

		EmulatedParticles_Thrust();
		void pushToDevice();
		void setupDev(EmulatedParticles_Dev * particles);
	};
}

#endif