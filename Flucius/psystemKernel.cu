#include "PSystem.h"

#include <glm\gtx\transform.hpp>
#include <glm\gtx\norm.hpp>
#include "device_launch_parameters.h" 
#include <thrust\device_ptr.h>
#include <thrust\detail\raw_pointer_cast.h>

#include "cudaHelper.h"
#include "neighbours.h"

#define ITERATIONS_COUNT 4
#define DELTA_Q (0.1f*PARTICLE_H)
#define PRESSURE_K 0.1
#define PRESSURE_N 6
#define REST_DENSITY 150000000.0f 

//KERNEL FUNCTIONS(in terms of math)
__device__ float wPoly6(glm::vec3 i, glm::vec3 j)
{
	float r = glm::length(i - j);
	return r > PARTICLE_H ? 0 : POLY6_CONST * powf(PARTICLE_H * PARTICLE_H - r * r, 3) / PARTICLE_H9;
}

__device__ glm::vec3 wGradSpiky(glm::vec3 i, glm::vec3 j)
{
	glm::vec3 r = i - j;
	float diff = PARTICLE_H - glm::length(r);
	float magnitude = (SPIKY_CONST / PARTICLE_H6) * diff * diff;
	return magnitude / (glm::length(r) + EPS) * r;
}

__device__ float calculateRo(Particle * particles, glm::vec3 curPos, int * neighbours, int nCnt, int id)
{
	float ro = 0.0f;
	for(int i = 0; i < nCnt; i++){
		ro += wPoly6(curPos, particles[neighbours[id * MAX_NEIGHBOURS + i]].pos);
	}
	return ro;
}

__device__ glm::vec3 cGradient(glm::vec3 i, glm::vec3 j){
	glm::vec3 Ci = -1.0f / float(REST_DENSITY) * wGradSpiky(i, j);
	return Ci;
}

__device__ glm::vec3 cGradientForI(Particle * particles, glm::vec3 curPos, int* neighbours, int nCnt, int id){
	glm::vec3 sum = glm::vec3(0.0f);
	for(int i = 0; i < nCnt; i++){
		sum += wGradSpiky(curPos, particles[neighbours[id * MAX_NEIGHBOURS + i]].pos);
	}
	glm::vec3 Ci = 1.0f / float(REST_DENSITY) * sum;
	return Ci;
}

//CUDA KERNELS

__global__ void calculateLambda(EmulatedParticles_Dev particles)
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, particles.count - 1);
	glm::vec3 curPos = particles.particles[particle].pos;
	int nCnt = particles.neighboursCnt[particle];

	float pI = calculateRo(particles.particles, curPos, particles.neighbours, nCnt, particle);
	float cI = (pI / REST_DENSITY) - 1.0f;


	float cIGradient, sumGradients = 0.0f;
	for (int i = 0; i < nCnt; i++) {
		//calc gradient with respect to other particle
		cIGradient = glm::length(cGradient(curPos, particles.particles[particles.neighbours[particle * MAX_NEIGHBOURS + i]].pos));
		sumGradients += (cIGradient * cIGradient);
	}

	//calc gradient with respect to current particle
	cIGradient = glm::length(cGradientForI(particles.particles, curPos, particles.neighbours, nCnt, particle));
	sumGradients += (cIGradient * cIGradient);

	float sumCi = sumGradients + RELAXATION;
	particles.particles[particle].lambda = - cI / sumCi; 
}

__global__ void calculateDeltaPos(EmulatedParticles_Dev particles)
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, particles.count - 1);
	int nCnt = particles.neighboursCnt[particle];
	glm::vec3 curPos = particles.particles[particle].pos;
	float lambda = particles.particles[particle].lambda;

	glm::vec3 delta = glm::vec3(0.0f);
	int other;
	float kTerm;
	glm::vec3 dq = DELTA_Q * glm::vec3(1.0f) + curPos;
	float sCorr = 0.0f;
	for (int i = 0; i < nCnt; i++) {
		other = particles.neighbours[i + particle * MAX_NEIGHBOURS];
		float poly6Pdq = wPoly6(curPos, dq);
		if (poly6Pdq < EPS) {
			kTerm = 0;
		} else {
			kTerm = wPoly6(curPos, glm::vec3(particles.particles[other].pos)) / poly6Pdq;
		}
		sCorr = -1.0f * PRESSURE_K * pow(kTerm, PRESSURE_N);
		delta += (lambda + particles.particles[other].lambda + sCorr) * wGradSpiky(curPos, glm::vec3(particles.particles[other].pos));
	}
	particles.particles[particle].deltaPos = delta / REST_DENSITY;
}


__global__ void applyExternalForces(EmulatedParticles_Dev particles, float dt) 
{
	int particle = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if (particle < particles.count) {
		int id = particles.particles[particle].id;
		particles.particles[particle].velocity += particles.externalForces[id] * dt;
		particles.particles[particle].pos = particles.prevPos[id] + dt * particles.particles[particle].velocity;
	}
}

__global__ void boxCollisionResponse(EmulatedParticles_Dev ep, Box box)
{
	int particle = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if( ep.particles[particle].pos.z < 0){
		ep.particles[particle].pos.z = EPS;
		glm::vec3 normal = glm::vec3(0.0f,0.0f,1.0f);
		glm::vec3 reflectedDir = ep.particles[particle].velocity - glm::vec3(2.0f*(normal*(glm::dot(ep.particles[particle].velocity,normal))));
		ep.particles[particle].velocity.z = reflectedDir.z;
	}
	if( ep.particles[particle].pos.z > box.size.z){
		ep.particles[particle].pos.z = box.size.z - EPS;
		glm::vec3 normal = glm::vec3(0.0f,0.0f,-1.0f);
		glm::vec3 reflectedDir = ep.particles[particle].velocity - glm::vec3(2.0f*(normal*(glm::dot(ep.particles[particle].velocity,normal))));
		ep.particles[particle].velocity.z = reflectedDir.z;
	}
	if( ep.particles[particle].pos.y < 0){
		ep.particles[particle].pos.y = EPS;
		glm::vec3 normal = glm::vec3(0.0f,1.0f,0.0f);
		glm::vec3 reflectedDir = ep.particles[particle].velocity - glm::vec3(2.0f*(normal*(glm::dot(ep.particles[particle].velocity,normal))));
		ep.particles[particle].velocity.y = reflectedDir.y;
	}
	if( ep.particles[particle].pos.y > box.size.y){
		ep.particles[particle].pos.y = box.size.y - EPS;
		glm::vec3 normal = glm::vec3(0.0f,-1.0f,0.0f);
		glm::vec3 reflectedDir = ep.particles[particle].velocity - glm::vec3(2.0f*(normal*(glm::dot(ep.particles[particle].velocity,normal))));
		ep.particles[particle].velocity.y = reflectedDir.y;
	}
	if( ep.particles[particle].pos.x < 0){
		ep.particles[particle].pos.x = EPS;
		glm::vec3 normal = glm::vec3(1.0f,0.0f,0.0f);
		glm::vec3 reflectedDir = ep.particles[particle].velocity - glm::vec3(2.0f*(normal*(glm::dot(ep.particles[particle].velocity,normal))));
		ep.particles[particle].velocity.x = reflectedDir.x;
	}
	if( ep.particles[particle].pos.x > box.size.x){
		ep.particles[particle].pos.x = box.size.x - EPS;
		glm::vec3 normal = glm::vec3(-1.0f,0.0f,0.0f);
		glm::vec3 reflectedDir = ep.particles[particle].velocity - glm::vec3(2.0f*(normal*(glm::dot(ep.particles[particle].velocity,normal))));
		ep.particles[particle].velocity.x = reflectedDir.x;
	}
}

__global__ void updatePredPositions(EmulatedParticles_Dev ep) 
{
	int particle = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if (particle < ep.count)
		ep.particles[particle].pos += ep.particles[particle].deltaPos;
}

__global__ void updateVelocities(EmulatedParticles_Dev ep, float dt) 
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, ep.count - 1);
	int id = ep.particles[particle].id;
	ep.particles[particle].velocity = (ep.particles[particle].pos - ep.prevPos[id]) / dt;
}

__global__ void updatePositions(EmulatedParticles_Dev ep) 
{
	int particle = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, ep.count - 1);
	int id = ep.particles[particle].id;
	ep.prevPos[id] = ep.particles[particle].pos;
}



void PSystem::update() {
	particles_t->setupDev(particles_dev);
	int count = particles_dev->count;
	float dt = 0.01f;

	applyExternalForces<<<getBlocks(count), getThreads(count)>>>(*particles_dev, dt);
	checkCudaErrorsWithLine("apply forces failed!");

	cudaFindKNeighbors(particles_dev->particles, particles_dev->count, partition3D, particles_dev->neighbours, particles_dev->neighboursCnt, box);	

	for (int i = 0; i < 5; i++) {
		calculateLambda<<<getBlocks(count), getThreads(count)>>>(*particles_dev);
		checkCudaErrorsWithLine("failed calculating lamdas!");

		calculateDeltaPos<<<getBlocks(count), getThreads(count)>>>(*particles_dev);
		checkCudaErrorsWithLine("failed calculating delta pos!");

		boxCollisionResponse<<<getBlocks(count), getThreads(count)>>>(*particles_dev, box);
		checkCudaErrorsWithLine("box collision failed!");

		updatePredPositions<<<getBlocks(count), getThreads(count)>>>(*particles_dev);
		checkCudaErrorsWithLine("update predPositions failed!");
	}
	updateVelocities<<<getBlocks(count), getThreads(count)>>>(*particles_dev, dt);	
	checkCudaErrorsWithLine("update velocities failed!");

	updatePositions<<<getBlocks(count), getThreads(count)>>>(*particles_dev);
	cudaDeviceSynchronize();

	/*int sum = 0;
	for (int i = 0; i < 50000; i++) {
		for (int j = 0; j < 9000; j++) {
			sum = sum * 15 - sum * sum * sum + 1;
		}
	}
	printf("%d", sum);*/

	glm::vec3 tPos = particles_t->positions[0];
	printf("%f, %f, %f\n", tPos.x, tPos.y, tPos.z);
	checkCudaErrorsWithLine("updatePositions failed!");

}