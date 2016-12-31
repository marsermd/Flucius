#include "PSystem.h"

#include <glm\gtx\norm.hpp>
#include "device_launch_parameters.h" 

#include "cudaHelper.h"
#include "neighbours.h"
#include "PSystemConstants.h"

using namespace SPH;

__device__ __constant__ Settings settings_dev[1];

//KERNEL FUNCTIONS(in terms of math)
__device__ float wPoly6(glm::vec3 i, glm::vec3 j)
{
	float r = glm::length(i - j);
	float tmp = PARTICLE_H * PARTICLE_H - r * r;
	return r > PARTICLE_H ? 0 : POLY6_CONST * tmp * tmp * tmp / PARTICLE_H9;
}

__device__ glm::vec3 wGradSpiky(glm::vec3 i, glm::vec3 j)
{
	glm::vec3 r = i - j;
	float length = glm::length(r);
	float diff = PARTICLE_H - length;
	if (diff < 0 || length < EPS)
	{
		return glm::vec3(0, 0, 0);
	}
	float magnitude = -(SPIKY_CONST / PARTICLE_H6) * diff * diff;
	return magnitude * r / length;
}

//CUDA KERNELS

__global__ void calculateLambda(EmulatedParticles_Dev particles)
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, particles.count - 1);
	if (particle < particles.count && !particles.particles[particle].isFixed)
	{
		glm::vec3 curPos = particles.particles[particle].pos;
		int nCnt = particles.neighboursCnt[particle];

		//Used to calculate CI
		float rhoI = 0.0f;

		// formula 8, part one
		glm::vec3 sumGradki = glm::vec3(0);
		//formula 8, part two
		float sumGradients = 0.0f;
		glm::vec3 gradient;
		for (int i = 0; i < nCnt; i++) 
		{
			glm::vec3 otherPos = particles.particles[particles.neighbours[particle * MAX_NEIGHBOURS + i]].pos;
			
			// Adding to RhoI to calculate CI
			rhoI += wPoly6(curPos, otherPos);

			gradient = wGradSpiky(curPos, otherPos) * settings_dev->oneOverRestDencity;
			//calc gradient with respect to other particle
			sumGradients += glm::length2(gradient);
			//calc gradient with respect to current particle
			sumGradki -= gradient;
		}
		sumGradients += glm::length2(sumGradki);

		float cI = (rhoI * settings_dev->oneOverRestDencity) - 1.0f;
		float sumCi = sumGradients + settings_dev->relaxation;
		particles.particles[particle].lambda = -cI / sumCi;
	}
}

__global__ void calculateDeltaPos(EmulatedParticles_Dev particles)
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, particles.count - 1);
	if (particle < particles.count && !particles.particles[particle].isFixed)
	{
		int nCnt = particles.neighboursCnt[particle];
		glm::vec3 curPos = particles.particles[particle].pos;
		float lambda = particles.particles[particle].lambda;

		glm::vec3 dq = settings_dev->deltaQ * glm::vec3(1.0f) + curPos;

		float tensileInstabilityScale = wPoly6(curPos, dq);
		if (tensileInstabilityScale < EPS) {
			//throw an error
			*(int*)0 = 0;
		}
		tensileInstabilityScale = 1.0f / tensileInstabilityScale;

		glm::vec3 delta = glm::vec3(0.0f);

		float kTerm, sCorr = 0.0f;
		int other;
		for (int i = 0; i < nCnt; i++) {
			other = particles.neighbours[i + particle * MAX_NEIGHBOURS];
			kTerm = wPoly6(curPos, glm::vec3(particles.particles[other].pos)) * tensileInstabilityScale;
			sCorr = -settings_dev->pressureK * kTerm * kTerm; // that means kterm ^ PRESSURE_N
			delta += (lambda + particles.particles[other].lambda + sCorr) * wGradSpiky(curPos, glm::vec3(particles.particles[other].pos));
		}
		particles.particles[particle].deltaPos = delta * settings_dev->oneOverRestDencity;
	}
}

__global__ void applyViscocityAndCalcVorticity(EmulatedParticles_Dev particles)
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, particles.count - 1);
	if (particle < particles.count && !particles.particles[particle].isFixed)
	{
		int nCnt = particles.neighboursCnt[particle];
		glm::vec3 curPos = particles.particles[particle].pos;
		glm::vec3 velocity = particles.particles[particle].velocity;

		glm::vec3 vAverage = glm::vec3(0);
		glm::vec3 vorticity = glm::vec3(0);
		for (int i = 0; i < nCnt; i++) 
		{
			Particle other = particles.particles[particles.neighbours[particle * MAX_NEIGHBOURS + i]];
			float wJ = wPoly6(curPos, other.pos);
			glm::vec3 deltaVIJ = other.velocity - velocity;

			vAverage += deltaVIJ * wJ;
			vorticity += glm::cross(deltaVIJ, wGradSpiky(curPos, other.pos));
		}

		velocity += vAverage * settings_dev->viscosity;

		particles.particles[particle].nextVelocity = velocity;
		particles.vorticities[particle] = vorticity;
	}
}

__global__ void applyVorticity(EmulatedParticles_Dev particles, float dt)
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, particles.count - 1);
	if (particle < particles.count && !particles.particles[particle].isFixed)
	{
		int nCnt = particles.neighboursCnt[particle];
		glm::vec3 curPos = particles.particles[particle].pos;
		glm::vec3 velocity = particles.particles[particle].velocity;

		glm::vec3 vorticityGradient = glm::vec3(0);
		for (int i = 0; i < nCnt; i++)
		{
			Particle other = particles.particles[particles.neighbours[particle * MAX_NEIGHBOURS + i]];
			vorticityGradient += glm::length(particles.vorticities[i]) * wGradSpiky(curPos, other.pos);
		}

		if (glm::length(vorticityGradient) > EPS)
		{
			vorticityGradient = glm::normalize(vorticityGradient);
		}

		glm::vec3 vorticity = particles.vorticities[particle];

		particles.particles[particle].nextVelocity += glm::cross(vorticityGradient, vorticity) * dt * settings_dev->vorticityEpsilon;
	}
}

__global__ void updateVelocityToNewVelocity(EmulatedParticles_Dev particles)
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, particles.count - 1);
	if (particle < particles.count)
	{
		particles.particles[particle].velocity = particles.particles[particle].nextVelocity;
	}
}


__global__ void applyExternalForces(EmulatedParticles_Dev particles, float dt)
{
	int particle = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if (particle < particles.count && !particles.particles[particle].isFixed) {
		int id = particles.particles[particle].id;
		particles.particles[particle].velocity += settings_dev->getGravity() * dt;
		particles.particles[particle].pos = particles.prevPos[id] + dt * particles.particles[particle].velocity;
	}
}

__global__ void boxCollisionResponse(EmulatedParticles_Dev ep, Box box)
{
	int particle = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if (particle < ep.count)
	{
		if (ep.particles[particle].pos.z < 0) {
			ep.particles[particle].pos.z *= -1;
			glm::vec3 normal = glm::vec3(0.0f, 0.0f, 1.0f);
			glm::vec3 reflectedDir = ep.particles[particle].velocity - 2.0f*glm::dot(ep.particles[particle].velocity, normal);
			ep.particles[particle].velocity.z = reflectedDir.z;
		}
		if (ep.particles[particle].pos.z > box.size.z) {
			ep.particles[particle].pos.z = 2 * box.size.z - ep.particles[particle].pos.z;
			glm::vec3 normal = glm::vec3(0.0f, 0.0f, -1.0f);
			glm::vec3 reflectedDir = ep.particles[particle].velocity - 2.0f*glm::dot(ep.particles[particle].velocity, normal);
			ep.particles[particle].velocity.z = reflectedDir.z;
		}
		if (ep.particles[particle].pos.y < 0) {
			ep.particles[particle].pos.y *= -1;
			glm::vec3 normal = glm::vec3(0.0f, 1.0f, 0.0f);
			glm::vec3 reflectedDir = ep.particles[particle].velocity - 2.0f*glm::dot(ep.particles[particle].velocity, normal);
			ep.particles[particle].velocity.y = reflectedDir.y;
		}
		if (ep.particles[particle].pos.y > box.size.y) {
			ep.particles[particle].pos.y = 2 * box.size.y - ep.particles[particle].pos.y;
			glm::vec3 normal = glm::vec3(0.0f, -1.0f, 0.0f);
			glm::vec3 reflectedDir = ep.particles[particle].velocity - 2.0f*glm::dot(ep.particles[particle].velocity, normal);
			ep.particles[particle].velocity.y = reflectedDir.y;
		}
		if (ep.particles[particle].pos.x < 0) {
			ep.particles[particle].pos.x *= -1;
			glm::vec3 normal = glm::vec3(1.0f, 0.0f, 0.0f);
			glm::vec3 reflectedDir = ep.particles[particle].velocity - 2.0f*glm::dot(ep.particles[particle].velocity, normal);
			ep.particles[particle].velocity.x = reflectedDir.x;
		}
		if (ep.particles[particle].pos.x > box.size.x) {
			ep.particles[particle].pos.x = 2 * box.size.x - ep.particles[particle].pos.x;
			glm::vec3 normal = glm::vec3(-1.0f, 0.0f, 0.0f);
			glm::vec3 reflectedDir = ep.particles[particle].velocity - 2.0f*glm::dot(ep.particles[particle].velocity, normal);
			ep.particles[particle].velocity.x = reflectedDir.x;
		}
	}
}

__global__ void updatePredPositions(EmulatedParticles_Dev ep)
{
	int particle = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if (particle < ep.count && !ep.particles[particle].isFixed)
		ep.particles[particle].pos += ep.particles[particle].deltaPos;
}

__global__ void updateVelocities(EmulatedParticles_Dev ep, float dt)
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, ep.count - 1);
	if (particle < ep.count)
	{
		int id = ep.particles[particle].id;
		ep.particles[particle].velocity = (ep.particles[particle].pos - ep.prevPos[id]) / dt;
	}
}

__global__ void updatePositions(EmulatedParticles_Dev ep)
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, ep.count - 1);
	if (particle < ep.count)
	{
		int id = ep.particles[particle].id;
		ep.prevPos[id] = ep.particles[particle].pos;
	}
}

void PSystem::update() {
	particles_t->setupDev(particles_dev);
	int count = particles_dev->count;
	float dt = 0.02f;

	cudaMemcpyToSymbol(settings_dev, &settings, sizeof(Settings));
	checkCudaErrorsWithLine("failed setting settings!");

	applyExternalForces << <getBlocks(count), getThreads(count) >> >(*particles_dev, dt);
	checkCudaErrorsWithLine("apply forces failed!");

	cudaFindKNeighbors(particles_dev->particles, particles_dev->count, partition3D, particles_dev->neighbours, particles_dev->neighboursCnt);

	for (int i = 0; i < settings.iterationsCount; i++) {
		calculateLambda << <getBlocks(count), getThreads(count) >> >(*particles_dev);
		checkCudaErrorsWithLine("failed calculating lamdas!");

		calculateDeltaPos << <getBlocks(count), getThreads(count) >> >(*particles_dev);
		checkCudaErrorsWithLine("failed calculating delta pos!");

		updatePredPositions << <getBlocks(count), getThreads(count) >> >(*particles_dev);
		checkCudaErrorsWithLine("update predPositions failed!");

		boxCollisionResponse << <getBlocks(count), getThreads(count) >> >(*particles_dev, box);
		checkCudaErrorsWithLine("box collision failed!");
	}
	updateVelocities << <getBlocks(count), getThreads(count) >> >(*particles_dev, dt);
	checkCudaErrorsWithLine("update velocities failed!");

	applyViscocityAndCalcVorticity << <getBlocks(count), getThreads(count) >> >(*particles_dev);
	checkCudaErrorsWithLine("apply voscocity failed!");

	applyVorticity << <getBlocks(count), getThreads(count) >> >(*particles_dev, dt);
	checkCudaErrorsWithLine("apply vorticity failed!");

	updateVelocityToNewVelocity << <getBlocks(count), getThreads(count) >> >(*particles_dev);
	checkCudaErrorsWithLine("update Velocity To New Velocity failed!");

	updatePositions << <getBlocks(count), getThreads(count) >> >(*particles_dev);
	cudaDeviceSynchronize();
	checkCudaErrorsWithLine("update positions failed!");
}

void PSystem::cudaInit()
{
	printf("initing psystem cuda");
	//cudaMalloc(&settings_dev, sizeof(Settings));
	//checkCudaErrorsWithLine("failed creating settings!");
}

void PSystem::cudaClear()
{
	printf("clearing psystem cuda");
	//cudaFree(&settings_dev);
	//checkCudaErrorsWithLine("failed freing settings!");
}