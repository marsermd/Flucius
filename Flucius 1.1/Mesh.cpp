#include "Mesh.h" 
#include <cstdio>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\transform.hpp>

void Mesh::bind(const Vertex* vertices, int vCount, const GLuint* indices, int iCount) {
	vertexCount = vCount;
	indexCount = iCount;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(2, vbo);

	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, vCount * sizeof(Vertex), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, iCount * sizeof(GLuint), indices, GL_STATIC_DRAW);

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

void Mesh::render() {
	if (!vao) {
		printf("Mesh object wasn't enabled");
	}
	uniformMatrices();

	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, NULL);
	glBindVertexArray(0);
}