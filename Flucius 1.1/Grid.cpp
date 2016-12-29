#include "Grid.h"
#include "PSystemConstants.h"

#include <glm\gtx\transform.hpp>

Grid::Grid(PSystem* pSystem) :
	count(120),
	box(pSystem->getBox())
{
	this->pSystem = pSystem;
	numVertices = (count + 1) * (count + 1) * (count + 1);
	numCubes = count * count * count;
	createVBO(numVertices);

	threshold = 1.0f;
	cudaInit(pSystem->getParticlesCount());
	partition3D = new Partition3D<GridVertex>(vertices_dev, numVertices, box, PARTICLE_H * 2.0f);
	cudaRestoreCVConnections();
}

Grid::~Grid()	{
	delete partition3D;

	deleteVBO();
	cudaClear();
};

void Grid::setCubeCount(int cnt) {
	cudaClear();
	count = cnt;
	cudaInit(pSystem->getParticlesCount());
}

void Grid::render() {
	modelMatrix = glm::translate(box.pos);
	
	cudaCalcGrid(pSystem->getParticlesPositions_dev(), pSystem->getParticlesCount());
	int totalVertices = cudaAnalyzeCubes(threshold);
	cudaComputeSurface(numVertices, threshold);

	glUniformMatrix4fv(modelMatrixID, 1, GL_FALSE, &modelMatrix[0][0]);

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, totalVertices);
	glBindVertexArray(0);
}

void Grid::createVBO(int size)
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

void Grid::deleteVBO()
{
	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);
	vbo = 0;
	glDeleteVertexArrays(1, &vao);
	vao = 0;
}
