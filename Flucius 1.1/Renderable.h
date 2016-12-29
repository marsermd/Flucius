#ifndef RENDERABLE_H
#define RENDERABLE_H
#include <gl\glew.h>
#include <glm\glm.hpp>

class Renderable{
public:
	virtual ~Renderable(){};

	virtual void render() = 0;
	//update uniform matrices
	virtual void uniformMatrices();

	GLuint modelMatrixID;
	glm::mat4 modelMatrix;
};

#endif