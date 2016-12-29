#ifndef SIMPLE_PSYSTEM_DRAWER_H
#define SIMPLE_PSYSTEM_DRAWER_H
#include <GL\glew.h>

#include "Partition3D.h"
#include "PSystem.h"
#include "Mesh.h"

/*
 * This class draws PSystem's particles as a set of points
 */
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

	void cudaUpdateVBO();

	void createVBO(int size);

	void deleteVBO();

	//_______________________________DEVICE VARIABLES________________________________________________________________________________________________________

	Vertex *triangleVertices_dev = 0;
	cudaGraphicsResource *cuda_vbo_resource;
};
#endif