#ifndef GRID_H
#define GRID_H
#include <GL\glew.h>
#include <glm\glm.hpp>

#include "Partition3D.h"
#include "Box.h"
#include "PSystem.h"
#include "Mesh.h"
#include "GridStructures.h"

/*
 * This class extracts and renders surface from PSystem using marching cubes
 */
class Grid: public Renderable {
public:
	Grid(PSystem* psystem);
	~Grid();

	virtual void render();  

	void setCubeCount(int cnt);//cube count per dimension
	
private:
	PSystem* pSystem;
	int numVertices;
	GridVertex *vertices_dev;

	int numCubes;
	GridCube* cubes_dev;

	Box box; // should always be a cube
	int count;//cube count per dimension
	float threshold;//threshold value to determine if vertex is active or not
	
	Partition3D<GridVertex>* partition3D;

	GLuint vbo, vao;

	int cudaCalcGrid(glm::vec3* particles_dev, size_t pCount);
	int cudaAnalyzeCubes(float threshold);
	void cudaComputeSurface(int maxVerts, float threshold);
	void cudaInit(size_t pCount);
	void cudaClear();
	void initGrid();
	void createCudaMemory(size_t pCount);
	void deleteCudaMemory();
	void allocateTextures();
	void cudaRestoreCVConnections();

	void createVBO(int size);
	void deleteVBO();
	//_______________________________DEVICE VARIABLES________________________________________________________________________________________________________

	int* cubesOccupied_dev = 0;
	int* cubesOccupiedScan_dev = 0;
	int* cubesCompact_dev = 0;
	int* cubeVerticesCnt_dev = 0;
	int* cubeVerticesScan_dev = 0;

	Vertex* triangleVertices_dev = 0;

	int* triTable_dev;
	int* vertsCountTable_dev;

	int* verticesOccupied_dev = 0;
	int* verticesOccupiedScan_dev = 0;
	int* verticesCompact_dev = 0;

	int* verticesToCubes_dev = 0;
	int* cubesToVertices_dev = 0;
	cudaGraphicsResource* cuda_vbo_resource;
};
#endif