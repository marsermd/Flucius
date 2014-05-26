#ifndef BOX_H
#define BOX_H
#define EPS 1.0E-5f
#include <GLFW\glfw3.h>
#include <glm\glm.hpp>
class Box {
public:
	Box (glm::vec3 pos, glm::vec3 size) : 
		pos(pos), 
		size(size) 
		{};
	Box (glm::vec3 pos, GLfloat size) : 
		pos(pos), 
		size(glm::vec3(size, size, size)) 
		{};
	glm::vec3 pos; //min x, min y, min z
	glm::vec3 size;
	bool contains(glm::vec3 vertex) {
		bool ans = true;
		for (int i = 0; i < 3; i++) {
			if (vertex[i] < pos[i] - EPS || vertex[i] > pos[i] + size[i] + EPS) {
				ans = false;
			}
		}
		return ans;
	}
};

#endif

