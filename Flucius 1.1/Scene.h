#ifndef SCENE_H
#define SCENE_H

#include <list>

#include "Camera.h"
#include "Mesh.h"

class Scene {
public:
	GLuint programID;
	Camera* camera;

	Scene(Camera* camera, GLuint programID) :
		camera(camera),
		programID(programID)
	{}

	// Render all scene objects
	void render();
	// Add object to render each frame
	void registerObject(Renderable* object);

private:
	std::list<Renderable*> objects;
};

#endif