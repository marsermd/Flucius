#ifndef GRID_STRUCTURES_H
#define GRID_STRUCTURES_H
#include <glm\glm.hpp>

struct GridVertex{
	glm::vec3 pos;
	glm::vec3 normal; //gradient of the scalar field
	float value;	//the value of the scalar field at this point
	int cubeID[8]; //Cube, related to current vertex(position of cube matches position of vertex in vID list)
};

struct GridCube {
	int vID[8];	//pointers to vertices(actually these are indecies, but who cares?)
	int cubeID;
};

#endif