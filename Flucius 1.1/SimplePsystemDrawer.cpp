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

void SimplePsystemDrawer::render() 
{
	modelMatrix = glm::mat4();

	updateParticlesCount();
	cudaUpdateVBO();

	glUniformMatrix4fv(modelMatrixID, 1, GL_FALSE, &modelMatrix[0][0]);

	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, lastParticlesCount);
	glBindVertexArray(0);
}

void SimplePsystemDrawer::createVBO(int size)
{
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// create buffer object
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size * sizeof(Vertex), 0, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
		sizeof(Vertex), (const GLvoid*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
		sizeof(Vertex), (const GLvoid*)(sizeof(float[3])));
	glEnableVertexAttribArray(1);

	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE,
		sizeof(Vertex), (const GLvoid*)(sizeof(float[3]) + sizeof(float[2])));
	glEnableVertexAttribArray(2);
}

void SimplePsystemDrawer::deleteVBO() 
{
	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);
	vbo = 0;
	glDeleteVertexArrays(1, &vao);
	vao = 0;
}
