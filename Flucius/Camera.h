#ifndef CAMERA_H
#define CAMERA_H

#include <GLFW\glfw3.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\transform.hpp>
#include "Timer.h"

class Camera {
public:
	Camera(GLFWwindow *window) {
		timer = new Timer();
		parentWindow = window;
		distance = 12;
		angleX = 0;
		angleY = 0;
		projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 100.0f);
		react();
	}
	~Camera() {
		delete timer;
	}
	
	void react();
	glm::mat4 getMatrix() {
		return matrix;
	}
	glm::vec3 getPosition();

private:
	Timer* timer;
	GLFWwindow* parentWindow;
	glm::mat4 matrix;
	glm::mat4 projection;
	glm::mat4 view;
	float distance;
	float angleX; // degrees
	float angleY; // degrees
};
#endif