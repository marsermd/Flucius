#include "SimplePsystemDrawer.h"

#include <math.h>
#include <glm\glm.hpp>
#include <glm\gtx\norm.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h" 

#include "cudaHelper.h"
#include "PSystem.h"

//_______________________________CUDA PART___________________________________________________________________________________________________________


__global__ void createParticleQuadsKernel(Particle* particles, int pCount, Vertex* vertices)
{
	int id = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if (id < pCount)
	{
#pragma unroll
		for (int j = 0; j < 3; j++) {
			vertices[id].position[j] = particles[id].pos[j];
			vertices[id].normal[j] = __saturatef((0.8 + particles[id].lambda * 50));
		}
	}
}

void SimplePsystemDrawer::cudaUpdateVBO()
{
	size_t num_bytes;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&triangleVertices_dev, &num_bytes, cuda_vbo_resource);
	checkCudaErrorsWithLine("failed setting up vbo");

	int pCount = pSystem->getParticlesCount();
	createParticleQuadsKernel << < getBlocks(pCount), getThreads(pCount) >> >(pSystem->getParticles_dev(), pCount, triangleVertices_dev);

	checkCudaErrorsWithLine("generate triangles failed");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	cudaDeviceSynchronize();
	checkCudaErrorsWithLine("failed unsetting vbo");
}

void SimplePsystemDrawer::cudaInit() {
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsRegisterFlagsNone));
}

void SimplePsystemDrawer::cudaClear() {

	cudaGraphicsUnregisterResource(cuda_vbo_resource);
}