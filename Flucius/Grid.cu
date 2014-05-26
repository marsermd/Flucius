#include "Grid.h"
#include "cudaHelper.h"

Grid::Grid(): 
	count(150),
	box(Box(glm::vec3(-3, -3, -3), 6.0f))
{
	numVertices = (count + 1) * (count + 1) * (count + 1);
	numCubes = count * count * count;
	createVBO(numVertices);

	threshold = 1;
	cudaInit(pSystem.getParticlesCount());
	partition3D = new Partition3D<GridVertex>(vertices_dev, numVertices, box, PARTICLE_R * 2.2f);
	cudaRestoreCVConnections();


	//int * partitions = cudaGetArr<int>(partition3D->partitions_dev, partition3D->ttlCount);
	//int * ptId = cudaGetArr<int>(partition3D->partitionIdx_dev, numVertices);
	//GridVertex * vtx = cudaGetArr<GridVertex>(vertices_dev, numVertices);
	//GridCube * cubes = cudaGetArr<GridCube>(cubes_dev, numCubes);
	//for (int i = 0; i < partition3D->ttlCount; i++){
	//	//pSystem.addParticle(vtx[partitions[i]].pos);
	//}
	//for (int i = 0; i < numVertices; i++) {
	//	int partitionId = ptId[i];
	//	if (partitions[partitionId] > i || (partitionId < partition3D->ttlCount - 1 && partitions[partitionId + 1] <= i)){
	//		printf("wtf? VERTEX_PARTITION");
	//	}
	//}
	//for (int i = 0; i < numVertices; i++) {
	//	for (int j = 0; j < 8; j++) {
	//		if (vtx[i].cubeID[j] >= 0 && cubes[vtx[i].cubeID[j]].vID[j] != i) {
	//			printf("wtf? CUBE-VERTEX");
	//		}
	//	}
	//	if (vtx[i].pos.x > box.size.x ||
	//		vtx[i].pos.y > box.size.y ||
	//		vtx[i].pos.z > box.size.z) {
	//			printf("wtf? vertex pos");
	//	}

	//	if (vtx[i].pos.x < 0 ||
	//		vtx[i].pos.y < 0 ||
	//		vtx[i].pos.z < 0) {
	//			printf("wtf? vertex pos");
	//	}
	//}

	//for (int i = 0; i < numCubes; i++) {
	//	glm::vec3 cVert[] = {
	//		vtx[cubes[i].vID[0]].pos,
	//		vtx[cubes[i].vID[1]].pos,
	//		vtx[cubes[i].vID[2]].pos,
	//		vtx[cubes[i].vID[3]].pos,
	//		vtx[cubes[i].vID[4]].pos,
	//		vtx[cubes[i].vID[5]].pos,
	//		vtx[cubes[i].vID[6]].pos,
	//		vtx[cubes[i].vID[7]].pos,
	//	};
	//	float max = box.size.x / count, min = box.size.x / count;
	//	max = std::max(glm::distance(cVert[0], cVert[1]), max);
	//	max = std::max(glm::distance(cVert[1], cVert[2]), max);
	//	max = std::max(glm::distance(cVert[2], cVert[3]), max);
	//	max = std::max(glm::distance(cVert[3], cVert[0]), max);
	//	max = std::max(glm::distance(cVert[4], cVert[5]), max);
	//	max = std::max(glm::distance(cVert[5], cVert[6]), max);
	//	max = std::max(glm::distance(cVert[6], cVert[7]), max);
	//	max = std::max(glm::distance(cVert[7], cVert[4]), max);
	//	max = std::max(glm::distance(cVert[0], cVert[4]), max);
	//	max = std::max(glm::distance(cVert[1], cVert[5]), max);
	//	max = std::max(glm::distance(cVert[2], cVert[6]), max);
	//	max = std::max(glm::distance(cVert[3], cVert[7]), max);

	//	min = std::min(glm::distance(cVert[0], cVert[1]), min);
	//	min = std::min(glm::distance(cVert[1], cVert[2]), min);
	//	min = std::min(glm::distance(cVert[2], cVert[3]), min);
	//	min = std::min(glm::distance(cVert[3], cVert[0]), min);
	//	min = std::min(glm::distance(cVert[4], cVert[5]), min);
	//	min = std::min(glm::distance(cVert[5], cVert[6]), min);
	//	min = std::min(glm::distance(cVert[6], cVert[7]), min);
	//	min = std::min(glm::distance(cVert[7], cVert[4]), min);
	//	min = std::min(glm::distance(cVert[0], cVert[4]), min);
	//	min = std::min(glm::distance(cVert[1], cVert[5]), min);
	//	min = std::min(glm::distance(cVert[2], cVert[6]), min);
	//	min = std::min(glm::distance(cVert[3], cVert[7]), min);

	//	if (max > box.size.x / count  + 0.0001 || min < box.size.x / count - 0.0001) {
	//		printf("WTF? cube-vertex pos %d \n", max);
	//	}
	//}

	//cudaFreeHost(ptId);
	//cudaFreeHost(partitions);
	//cudaFreeHost(vtx);
	//cudaFreeHost(cubes);
	//cudaDeviceSynchronize();
	//checkCudaErrorsWithLine("try to sync");

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