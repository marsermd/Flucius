#ifndef CAMERA_H
#define CAMERA_H

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm.hpp>
#include "Timer.h"

class Camera {
public:
	/*
	 * Camera listen's to window's events
	 * It rotates around the center at given distance
	 */
	Camera(GLFWwindow *window, glm::vec3 center, float distance);
	~Camera();
	
	// Update transform according to user input
	void react();

	// Get ViewProjection matrix
	glm::mat4 getMatrix();

	/*
	 * Returns camera's position
	 */
	glm::vec3 getPosition();

private:
	Timer* timer;
	GLFWwindow* parentWindow;
	glm::mat4 matrix;
	glm::mat4 projection;
	glm::mat4 view;
	glm::vec3 center;
	float distance;
	float angleX; // degrees
	float angleY; // degrees
};
#endif