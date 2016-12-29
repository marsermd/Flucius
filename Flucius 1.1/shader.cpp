#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#include <stdlib.h>

#include <GL/glew.h>

#include "shader.h"


void loadShader(GLuint * shaderID, const char * file_path, GLuint programID) {
	// Read shader
	std::string shaderCode;
	std::ifstream shaderStream(file_path, std::ios::in);
	if (shaderStream.is_open()) {
		std::string Line = "";
		while(getline(shaderStream, Line))
			shaderCode += "\n" + Line;
		shaderStream.close();
	} else {
		printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", file_path);
		//getchar();
		exit(-1);
	}

	// Compile shader
	printf("Compiling shader : %s\n", file_path);
	char const * sourcePointer = shaderCode.c_str();
	glShaderSource(*shaderID, 1, &sourcePointer , NULL);
	glCompileShader(*shaderID);

	GLint result = GL_FALSE;
	int infoLogLength;

	// check shader
	glGetShaderiv(*shaderID, GL_COMPILE_STATUS, &result);
	glGetShaderiv(*shaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
	if (infoLogLength > 0) {
		std::vector<char> shaderErrorMessage(infoLogLength+1);
		glGetShaderInfoLog(*shaderID, infoLogLength, NULL, &shaderErrorMessage[0]);
		printf("%s\n", &shaderErrorMessage[0]);
	}

	glAttachShader(programID, *shaderID);
	glLinkProgram(programID);

	// Check the program
	glGetProgramiv(programID, GL_LINK_STATUS, &result);
	glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);
	if (infoLogLength > 0) {
		std::vector<char> programErrorMessage(infoLogLength+1);
		glGetProgramInfoLog(programID, infoLogLength, NULL, &programErrorMessage[0]);
		printf("%s\n", &programErrorMessage[0]);
	}
}

GLuint loadShaders(const char * vertexFilePath, const char * fragmentFilePath, const char * geometryFilePath) {

	// Create the shaders
	GLuint programID = glCreateProgram();

	GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	loadShader(&vertexShaderID, vertexFilePath, programID);
	glDeleteShader(vertexShaderID);

	GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
	loadShader(&fragmentShaderID, fragmentFilePath, programID);
	glDeleteShader(fragmentShaderID);

	if (geometryFilePath != NULL)
	{
		GLuint geometryShaderID = glCreateShader(GL_GEOMETRY_SHADER);
		loadShader(&geometryShaderID, geometryFilePath, programID);
		glDeleteShader(geometryShaderID);
	}


	return programID;
}


