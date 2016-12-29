#include "Camera.h"
#include <glm/gtc/matrix_transform.inl>

Camera::Camera(GLFWwindow* window, glm::vec3 center, float distance) :
	center(center),
	distance(distance)
{
	timer = new Timer();
	parentWindow = window;
	angleX = 0;
	angleY = 0;
	projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 500.0f);
	react();
}

Camera::~Camera()
{
	delete timer;
}

glm::mat4 Camera::getMatrix()
{
	return matrix;
}

glm::vec3 Camera::getPosition() {
	float aX = glm::radians(angleX);
	float aY = glm::radians(angleY);
	glm::vec3 pos
		( 
			glm::sin(aX) * glm::cos(aY) * distance,
			glm::sin(aY) * distance, 
			glm::cos(aX) * glm::cos(aY) * distance
		);
	return pos + center;
}

void Camera::react() {
	const float speed = 100;//degrees per second
	if (glfwGetKey(parentWindow, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		angleX += speed * timer->getDelta();
	}
	if (glfwGetKey(parentWindow, GLFW_KEY_LEFT) == GLFW_PRESS) {
		angleX -= speed * timer->getDelta();
	}
	if (glfwGetKey(parentWindow, GLFW_KEY_UP) == GLFW_PRESS) {
		if (angleY < 80) {
			angleY += speed * timer->getDelta();
		}
	}
	if (glfwGetKey(parentWindow, GLFW_KEY_DOWN) == GLFW_PRESS) {
		if (angleY > -80) {
			angleY -= speed * timer->getDelta();
		}
	}
	timer -> step();

	view = glm::lookAt
		(
			getPosition(),
			center,
			glm::vec3(0,1,0)
		);
	matrix = projection * view;
}