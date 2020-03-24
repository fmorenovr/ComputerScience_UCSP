
#include <GL/glew.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>

#include "gizmo.h"

#include "common/shader.hpp"
#include "common/texture.hpp"
#include "common/controls.hpp"
#include "common/objloader.hpp"
#include "common/vboindexer.hpp"



Gizmo::Gizmo(){
	glGenVertexArrays(1, &VertexArrayID);
	ModelMatrix = glm::mat4(1.0);

	vertices.push_back(glm::vec3(0.0,0.0,0.0));
	vertices.push_back(glm::vec3(2.0,0.0,0.0));
	
	vertices.push_back(glm::vec3(0.0,0.0,0.0));
	vertices.push_back(glm::vec3(0.0,2.0,0.0));
	
	vertices.push_back(glm::vec3(0.0,0.0,0.0));
	vertices.push_back(glm::vec3(0.0,0.0,2.0));

	colors.push_back(glm::vec3(255.0,0.0,0.0));
	colors.push_back(glm::vec3(255.0,0.0,0.0));


	colors.push_back(glm::vec3(0.0,255.0,0.0));
	colors.push_back(glm::vec3(0.0,255.0,0.0));

	colors.push_back(glm::vec3(0.0,0.0,255.0));
	colors.push_back(glm::vec3(0.0,0.0,255.0));

	glBindVertexArray(VertexArrayID);
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_STATIC_DRAW);



}


Gizmo::~Gizmo(){

	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &colorbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);


}

void Gizmo::setShaders(std::string vfile,std::string ffile){


	programID = LoadShaders( vfile.c_str(), ffile.c_str());

	// Get a handle for our "MVP" uniform
	MatrixID = glGetUniformLocation(programID, "MVP");


}




void Gizmo::draw(){

	glUseProgram(programID);
	glm::mat4 ProjectionMatrix = getProjectionMatrix();
	glm::mat4 ViewMatrix = getViewMatrix();

	glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader, 
		// in the "MVP" uniform
	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
			);

		// 2nd attribute buffer : UVs
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glVertexAttribPointer(
			1,                                // attribute
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
			);

		// 3rd attribute buffer : normals
	
	glDrawArrays(GL_LINES, 0, vertices.size() );

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);




}
