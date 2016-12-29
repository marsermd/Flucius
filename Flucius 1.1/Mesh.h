#ifndef MESH_H
#define MESH_H
#include "Renderable.h"

#include <gl\glew.h>

#include "Vertex.h"

/*
 * Realisation of renderable that renders Vertex* vertices
 */
class Mesh : public Renderable
{
public:	
	/*
	 * vCount vertices stored in vertices
	 * iCount indices stored in indices
	 */
	void bind(const Vertex* vertices, int vCount, const GLuint* indices, int iCount);
	virtual void render();

private:
	int indexCount;
	int vertexCount;
	GLuint vbo[2];
	GLuint vao;
};

#endif