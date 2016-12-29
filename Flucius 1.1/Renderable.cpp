#include "Renderable.h"

void Renderable::uniformMatrices()
{
	glUniformMatrix4fv(modelMatrixID, 1, GL_FALSE, &modelMatrix[0][0]);
}
