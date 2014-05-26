#ifndef MESH_H
#define MESH_H
#include "Renderable.h"

#include <gl\glew.h>
#include <glm\glm.hpp>

#include "Vertex.h"

class Mesh : public Renderable{
public:	
	void bind(const Vertex * vertices, int vCount, const GLuint * indices, int iCount);
	virtual void render();

private:
	int indexCount;
	int vertexCount;
	GLuint vbo[2];
	GLuint vao;
};

#endif