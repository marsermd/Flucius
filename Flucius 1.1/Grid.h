#ifndef GRID_H
#define GRID_H
#include <GL\glew.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\transform.hpp>

#include "Partition3D.h"
#include "Box.h"
#include "PSystem.h"
#include "Mesh.h"
#include "GridStructures.h"

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

	void createVBO(int size) {
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

	void deleteVBO() {
		glBindBuffer(1, vbo);
		glDeleteBuffers(1, &vbo);
		vbo = 0;
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}
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