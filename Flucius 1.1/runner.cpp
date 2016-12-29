#include <stdio.h>
#include <stdlib.h>

#include <GL\glew.h>

#include <GLFW\glfw3.h>

#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>

#include <cuda_profiler_api.h>

#include "shader.h"
#include "Camera.h"
#include "Grid.h"
#include "SimplePsystemDrawer.h"
#include "Scene.h"

struct Material
{
	glm::vec3 color;
	glm::vec4 ambient;
	glm::vec4 diffuse;
	glm::vec4 specular;
	glm::vec4 emission;
	GLfloat shininess;
};

void MaterialSetup(GLuint program, const Material &material)
{
	GLuint colorID = glGetUniformLocation(program, "material.color");
	glUniform3f(colorID, 0.0f, 0.0f, 0.8f);

	glUniform4fv(glGetUniformLocation(program, "material.ambient"),  1, glm::value_ptr(material.ambient));
	glUniform4fv(glGetUniformLocation(program, "material.diffuse"),  1, glm::value_ptr(material.diffuse));
	glUniform4fv(glGetUniformLocation(program, "material.specular"), 1, glm::value_ptr(material.specular));
	glUniform4fv(glGetUniformLocation(program, "material.emission"), 1, glm::value_ptr(material.emission));

	glUniform1fv(glGetUniformLocation(program, "material.shininess"), 1, &material.shininess);
}

struct PointLight
{
	glm::vec4  position;
	glm::vec4  ambient;
	glm::vec4  diffuse;
	glm::vec4  specular;
	glm::vec3  attenuation;
};

void PointLightSetup(GLuint program, const PointLight &light)
{
	glUniform4fv(glGetUniformLocation(program, "light.position"),    1, glm::value_ptr(light.position));
	glUniform4fv(glGetUniformLocation(program, "light.ambient"),     1, glm::value_ptr(light.ambient));
	glUniform4fv(glGetUniformLocation(program, "light.diffuse"),     1, glm::value_ptr(light.diffuse));
	glUniform4fv(glGetUniformLocation(program, "light.specular"),    1, glm::value_ptr(light.specular));
	glUniform3fv(glGetUniformLocation(program, "light.attenuation"), 1, glm::value_ptr(light.attenuation));
}

GLFWwindow* window;

void openWindow(int width, int height) {
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		exit(-1);
	}

	glfwWindowHint(GLFW_SAMPLES, 2); // 2x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	//GLFWmonitor * primary = glfwGetPrimaryMonitor();
	

	window = glfwCreateWindow(width, height, "Flucius", NULL, NULL);
	
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental=true;
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		exit(-1);
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
}

void updatePsystemSettings(PSystem& pSystem, GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		pSystem.settings.setGravity(glm::vec3(0, 0, -10));
	}
	else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		pSystem.settings.setGravity(glm::vec3(0, 0, 10));
	}
	else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		pSystem.settings.setGravity(glm::vec3(-10, 0, 0));
	}
	else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		pSystem.settings.setGravity(glm::vec3(10, 0, 0));
	}
	else
	{
		pSystem.settings.setGravity(glm::vec3(0, -10, 0));
	}

	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		pSystem.settings.viscosity = glm::clamp(pSystem.settings.viscosity - 0.02f, 0.0f, 1.0f);
	}
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
	{
		pSystem.settings.viscosity = glm::clamp(pSystem.settings.viscosity + 0.02f, 0.0f, 1.0f);
	}

	if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
	{
		pSystem.settings.iterationsCount = glm::clamp(pSystem.settings.iterationsCount - 1, 1, 10);
	}
	if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS)
	{
		pSystem.settings.iterationsCount = glm::clamp(pSystem.settings.iterationsCount + 1, 1, 10);
	}
}

int main() {
	openWindow(1024, 768);

	Material material;
	material.color = glm::vec3(1.0, 1.0, 0.5);
	material.ambient = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f);
	material.diffuse = glm::vec4(0.3f, 0.5f, 1.0f, 1.0f);
	material.specular = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
	material.emission = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	material.shininess = 20.0f;

	PointLight pointLight;
	pointLight.position = glm::vec4(0.0f, 0.0f, 7.0f, 1.0f);
	pointLight.ambient = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
	pointLight.diffuse = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	pointLight.specular = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	pointLight.attenuation = glm::vec3(0.2f, 0.0f, 0.02f);

	GLuint programID = loadShaders("TransformVertexShader.vertexshader", "ColorFragmentShader.fragmentshader");

	//display range : 0.1 unit <-> 100 units
	Camera camera(window, glm::vec3(35, 20, 35), 130);

	PSystem pSystem = PSystem(70.0f);
	SimplePsystemDrawer pSystemDrawer = SimplePsystemDrawer(&pSystem);
	//Grid pSystemGrid = Grid(&pSystem);
	pSystem.setRenderer(&pSystemDrawer);

	Scene scene(&camera, programID);
	scene.registerObject(&pSystem);

	Timer fpsTimer; 
	int frame = 0;

	bool wasSpacePressed = false;

	do {
		camera.react();
		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);

		glUseProgram(programID);

		MaterialSetup(programID, material);
		PointLightSetup(programID, pointLight);

		scene.render();

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
		frame = (frame + 1) % 10;
		if (frame == 0) {
			printf("fps = %.2f\n", 10.0f / fpsTimer.getDelta());
			fpsTimer.step();
		}

		updatePsystemSettings(pSystem, window);

		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		{
			if (!wasSpacePressed)
			{
				pSystem.addParticleBox(glm::vec3(20, 50, 20), 10);
			}
			wasSpacePressed = true;
		}
		else
		{
			wasSpacePressed = false;
		}

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);
	cudaProfilerStop();
}
