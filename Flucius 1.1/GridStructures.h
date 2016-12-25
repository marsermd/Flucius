#ifndef GRID_STRUCTURES_H
#define GRID_STRUCTURES_H
#include <glm\glm.hpp>

struct __align__(32) GridVertex{
	glm::vec3 pos;
	glm::vec3 normal; //gradient of the scalar field
	float value;	//the value of the scalar field at this point
	int id;
};

struct GridCube {
	int cubeID;
};

#endif