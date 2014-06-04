#include "Grid.h"
#include "cudaHelper.h"

Grid::Grid(): 
	count(120),
	box(Box(glm::vec3(-3, -3, -3), 6.0f))
{
	numVertices = (count + 1) * (count + 1) * (count + 1);
	numCubes = count * count * count;
	createVBO(numVertices);

	threshold = 1;
	cudaInit(pSystem.getParticlesCount());
	partition3D = new Partition3D<GridVertex>(vertices_dev, numVertices, box, PARTICLE_R * 2.2f);
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
	cudaInit(pSystem.getParticlesCount());
}

void Grid::render() {
	pSystem.update();
	modelMatrix = glm::translate(box.pos);
	cudaCalcGrid(pSystem.getParticles_dev(), pSystem.getParticlesCount());
	int totalVertices = cudaAnalyzeCubes(threshold);
	cudaComputeSurface(numVertices, threshold);

	glUniformMatrix4fv(modelMatrixID, 1, GL_FALSE, &modelMatrix[0][0]);

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, totalVertices);
	glBindVertexArray(0);
}