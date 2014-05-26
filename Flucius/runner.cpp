#include <stdio.h>
#include <stdlib.h>

#include <GL\glew.h>

#include <GLFW\glfw3.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\transform.hpp>
#include <glm\gtc\type_ptr.hpp>


#include "shader.h"
#include "Camera.h"
#include "Grid.h"
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
	// установка цвета
	GLuint colorID = glGetUniformLocation(program, "material.color");
	glUniform3f(colorID, 0.8f, 0.0f, 0.0f);

	// установка параметров
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
	// установка параметров
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

	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
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

#include "torus.h"
GLuint vao;

GLuint triangleVao;
void bindTriangle() {
	GLuint vbo[2];
	glGenVertexArrays(1, &triangleVao);
	glBindVertexArray(triangleVao);

	glGenBuffers(2, vbo);

	const Vertex triangleVertices[3] = {
		{{0.0f / size, 0.0f / size, 0.0f / size}, {0.500000f, 0.308658f}, {0.0f, 0.0f, 1.0f}},
		{{1.0f / size, 0.0f / size, 0.0f / size}, {0.500000f, 0.308658f}, {0.0f, 0.0f, 1.0f}},
		{{0.0f / size, 1.0f / size, 0.0f / size}, {0.500000f, 0.308658f}, {0.0f, 0.0f, 1.0f}}};

	const GLuint triangleIndices[3] = {0, 1, 2};

	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(Vertex), triangleVertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * sizeof(GLuint), triangleIndices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
		sizeof(Vertex), (const GLvoid*)0);

	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
		sizeof(Vertex), (const GLvoid*)(sizeof(float[3])));

	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE,
		sizeof(Vertex), (const GLvoid*)(sizeof(float[3]) + sizeof(float[2])));
}

void drawTriangle() {
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glBindVertexArray(triangleVao);
	glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);
}

glm::mat3 strip(glm::mat4 m) {
	return glm::mat3
		(
		m[0][0], m[0][1], m[0][2],
		m[1][0], m[1][1], m[1][2],
		m[2][0], m[2][1], m[2][2]
	);
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

	GLuint programID = LoadShaders("TransformVertexShader.vertexshader", "ColorFragmentShader.fragmentshader");

	//display range : 0.1 unit <-> 100 units
	Camera camera(window);

	bindTriangle();
	Mesh thor = Mesh();
	thor.bind(vertices, vcount, indices, icount);
	Grid grid = Grid();


	float a = 0;

	Timer fpsTimer; 
	int frame = 0;

	do {
		camera.react();
		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);

		glUseProgram(programID);

		GLuint viewProjectionMatrixID = glGetUniformLocation(programID, "transform.viewProjection");
		GLuint modelMatrixID = glGetUniformLocation(programID, "transform.model");
		GLuint cameraPositionID = glGetUniformLocation(programID, "transform.cameraPosition");

		glm::mat4 viewProjection = camera.getMatrix();
		a += 1;
		glm::vec3 cameraPosition = camera.getPosition();

		glUniformMatrix4fv(viewProjectionMatrixID, 1, GL_FALSE, &viewProjection[0][0]);
		glUniform3fv(cameraPositionID, 1, &cameraPosition[0]);

		MaterialSetup(programID, material);
		PointLightSetup(programID, pointLight);

		grid.modelMatrixID = modelMatrixID;
		grid.render();

		glm::mat4 initial = glm::translate(glm::scale(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f));
		glm::mat4 model = glm::rotate(glm::rotate(initial, a, glm::vec3(0.0f, 1.0f, 0.0f)), 45.0f, glm::vec3(-1.0f, 0.0f, 0.0f));

		thor.modelMatrixID = modelMatrixID;
		thor.modelMatrix = model;
		//thor.render();

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
		frame = (frame + 1) % 10;
		if (frame == 0) {
			printf("fps = %.2f\n", fpsTimer.getDelta() / 10.0f);
			fpsTimer.step();
		}

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);
}

