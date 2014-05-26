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
	Grid();
	~Grid();


	virtual void render();  

	void setCubeCount(int cnt);//cube count per dimension
	
	PSystem pSystem;
private:
	int numVertices;
	GridVertex *vertices_dev;

	int numCubes;
	GridCube *cubes_dev;

	Box box; // should always be a cube
	int count;//cube count per dimension
	float threshold;//threshold value to determine if vertex is active or not
	
	Partition3D<GridVertex> * partition3D;

	GLuint vbo, vao;

	int cudaCalcGrid(Particle * particles, int pCount);
	int cudaAnalyzeCubes(float threshold);
	void cudaComputeSurface(int maxVerts, float threshold);
	void cudaInit(int pCount);
	void cudaRestoreCVConnections();
	void cudaClear();
	void initGrid();
	void createCudaMemory(int pCount);
	void deleteCudaMemory();
	void allocateTextures();

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

#ifdef __TESTING__
#include "GridTest.h"
#include "Partition3D.h"
	friend class GridTest;
	friend class PartitionTest;
#endif
};
#endif