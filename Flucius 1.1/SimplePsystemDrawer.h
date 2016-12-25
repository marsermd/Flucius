#ifndef SIMPLE_PSYSTEM_DRAWER_H
#define SIMPLE_PSYSTEM_DRAWER_H
#include <GL\glew.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\transform.hpp>

#include "Partition3D.h"
#include "PSystem.h"
#include "Mesh.h"

class SimplePsystemDrawer : public Renderable {
public:
	SimplePsystemDrawer(PSystem* psystem);
	~SimplePsystemDrawer();

	virtual void render();

private:
	PSystem* pSystem;
	GLuint vbo, vao;

	int lastParticlesCount;
	int verticesCount;
	void updateParticlesCount();//cube count per dimension

	void cudaInit();
	void cudaClear();
	void createCudaMemory();
	void deleteCudaMemory();

	void cudaUpdateVBO();

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

	Vertex *triangleVertices_dev = 0;
	struct cudaGraphicsResource *cuda_vbo_resource;
};
#endif