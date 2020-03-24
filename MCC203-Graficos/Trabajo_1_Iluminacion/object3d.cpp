
#include <GL/glew.h>


#include <glm/gtc/matrix_transform.hpp>


#include <iostream>

#include "object3d.h"

#include "skeleton.h"

#include "common/shader.hpp"
#include "common/texture.hpp"
#include "common/controls.hpp"
#include "common/objloader.hpp"
#include "common/vboindexer.hpp"



Object3D::Object3D(){
	glGenVertexArrays(1, &VertexArrayID);
	Texture=0;
	ModelMatrix = glm::mat4(1.0);
	lightPos=glm::vec4(4,4,1,0);
}

Object3D::Object3D(std::string filename, float scale){
	glGenVertexArrays(1, &VertexArrayID);
	Texture=0;
	ModelMatrix = glm::mat4(1.0);
	lightPos=glm::vec4(4,4,1,0);


	
	meshFilename = filename;
	load_scale=scale;
	loadMesh();

}

Object3D::~Object3D(){

	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteBuffers(1, &normalbuffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &Texture);
	glDeleteVertexArrays(1, &VertexArrayID);
	delete skel;


}

void Object3D::setShaders(std::string vfile,std::string ffile){


	programID = LoadShaders( vfile.c_str(), ffile.c_str());

	// Get a handle for our "MVP" uniform
	MatrixID = glGetUniformLocation(programID, "MVP");
	ViewMatrixID = glGetUniformLocation(programID, "V");
	ModelMatrixID = glGetUniformLocation(programID, "M");
	TextureID  = glGetUniformLocation(programID, "myTextureSampler");
	LightID = glGetUniformLocation(programID, "LightPosition_worldspace");

}

void Object3D::loadMesh(){

	skel = new Skeleton();
	
	loadFBX(meshFilename.c_str(), vertices, uvs, normals,skel, load_scale);

	//Si no consiguen a compilar con el SDK FBX
	//loadOBJ(meshFilename.c_str(), vertices, uvs, normals);

	//skel->printSkeleton();
	/*skel->skeletonMesh();
	skel->skeletonVBO();
	skel->setShaders("gizmo.vertexshader", "gizmo.fragmentshader");
	std::cout << "finish loading" << std::endl;*/

	glBindVertexArray(VertexArrayID);
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);

	glGenBuffers(1, &normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), &normals[0], GL_STATIC_DRAW);

}


void Object3D::draw(){

	//skel->ModelMatrix = glm::translate(ModelMatrix,glm::vec3(0,0,1));
	//skel->draw();

	glUseProgram(programID);
	glm::mat4 ProjectionMatrix = getProjectionMatrix();
	glm::mat4 ViewMatrix = getViewMatrix();

	glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader, 
		// in the "MVP" uniform
	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
	glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);
	glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &ViewMatrix[0][0]);

	glm::mat4 lrot = glm::rotate(glm::mat4(1.0),0.0f,glm::vec3(0,1,0));

	lightPos =  lrot * lightPos;

	static int factor= 0.0;
	lightPos.x += 0.1*factor;

	if(lightPos.x>7.0){
		factor=-1;
	}
	else if(lightPos.x<-7.0){
		factor=1;
	}


	glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);

		// Bind our texture in Texture Unit 0
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Texture);
		// Set our "myTextureSampler" sampler to use Texture Unit 0
	glUniform1i(TextureID, 0);

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
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glVertexAttribPointer(
			1,                                // attribute
			2,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
			);

		// 3rd attribute buffer : normals
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glVertexAttribPointer(
			2,                                // attribute
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
			);

		// Draw the triangles !
	glDrawArrays(GL_TRIANGLES, 0, vertices.size() );

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);




}