#ifndef SCENE_H
#define SCENE_H


#include <list>

#include "Camera.h"
#include "Mesh.h"

class Scene {
public:
	GLuint programID;
	void render() {
		GLuint viewProjectionMatrixID = glGetUniformLocation(programID, "transform.viewProjection");
		GLuint modelMatrixID = glGetUniformLocation(programID, "transform.model");
		GLuint cameraPositionID = glGetUniformLocation(programID, "transform.cameraPosition");

		for (std::list<Mesh*>::iterator it = objects.begin(); it != objects.end(); ++it) {
			(*it)->modelMatrixID = modelMatrixID;
			(*it)->render();
		}
	}

	Camera camera;
	std::list<Mesh*> objects;
};

#endif