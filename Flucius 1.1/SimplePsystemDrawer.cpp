#include "SimplePsystemDrawer.h"
#include "cudaHelper.h"
#include "PSystemConstants.h"

SimplePsystemDrawer::SimplePsystemDrawer(PSystem* pSystem)
{
	this->pSystem = pSystem;
	createVBO(pSystem->getParticlesCount() * 6);

	cudaInit();
}

SimplePsystemDrawer::~SimplePsystemDrawer()	
{
	deleteVBO();
	cudaClear();
};


void SimplePsystemDrawer::updateParticlesCount() 
{
	int curParticleCount = pSystem->getParticlesCount();
	if (curParticleCount > lastParticlesCount)
	{
		lastParticlesCount = pSystem->getParticlesCount();
		cudaClear();
		deleteVBO();

		createVBO(pSystem->getParticlesCount() * 6);
		cudaInit();
	}
}

void SimplePsystemDrawer::render() {
	modelMatrix = glm::mat4();//identity

	updateParticlesCount();
	cudaUpdateVBO();

	glUniformMatrix4fv(modelMatrixID, 1, GL_FALSE, &modelMatrix[0][0]);

	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, lastParticlesCount);
	glBindVertexArray(0);
}