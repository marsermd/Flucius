#include "Camera.h"

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