#ifndef RENDERABLE_H
#define RENDERABLE_H
#include <gl\glew.h>
#include <glm\glm.hpp>

class Renderable{
public:
	virtual ~Renderable(){};

	virtual void render() = 0;
	virtual void uniformMatrices() {
		glUniformMatrix4fv(modelMatrixID, 1, GL_FALSE, &modelMatrix[0][0]);
	}

	GLuint modelMatrixID;
	glm::mat4 modelMatrix;
};

#endif