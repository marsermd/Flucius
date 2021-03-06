﻿#include "Scene.h"

void Scene::render()
{
	GLuint viewProjectionMatrixID = glGetUniformLocation(programID, "transform.viewProjection");
	GLuint modelMatrixID = glGetUniformLocation(programID, "transform.model");
	GLuint cameraPositionID = glGetUniformLocation(programID, "transform.cameraPosition");

	glm::mat4 viewProjection = camera->getMatrix();
	glm::vec3 cameraPosition = camera->getPosition();

	glUniformMatrix4fv(viewProjectionMatrixID, 1, GL_FALSE, &viewProjection[0][0]);
	glUniform3fv(cameraPositionID, 1, &cameraPosition[0]);

	for (std::list<Renderable*>::iterator it = objects.begin(); it != objects.end(); ++it) {
		(*it)->modelMatrixID = modelMatrixID;
		(*it)->render();
	}
}

void Scene::registerObject(Renderable* object)
{
	objects.push_back(object);
}
