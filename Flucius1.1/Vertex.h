#ifndef VERTEX_H
#define VERTEX_H
#include <gl\glew.h>

class Vertex
{
public:
	GLfloat position[3];
	GLfloat texcoord[2];
	GLfloat normal[3];

	void setPosition(const GLfloat * pos) {
		for (int i = 0; i < 3; i++) 
			position[i] = pos[i];
	}

	void setTexcoord(const GLfloat * tex) {
		for (int i = 0; i < 2; i++) 
			texcoord[i] = tex[i];
	}

	void setNormal(const GLfloat * norm) {
		for (int i = 0; i < 3; i++) 
			normal[i] = norm[i];
	}
};
#endif