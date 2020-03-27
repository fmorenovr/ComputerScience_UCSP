#include <GL/glew.h>


#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext.hpp>

#include <iostream>

#include "skeleton.h"
#include "node.h"

#include "common/shader.hpp"
#include "common/texture.hpp"
#include "common/controls.hpp"
#include "common/objloader.hpp"
#include "common/vboindexer.hpp"



Skeleton::Skeleton(){
	glGenVertexArrays(1, &VertexArrayID);
	ModelMatrix = glm::mat4(1.0);

	root = new Node(NULL,"init");
	root->isRoot = true;

}


Skeleton::~Skeleton(){

	delete root;

	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &colorbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);

}

void Skeleton::setShaders(std::string vfile,std::string ffile){


	programID = LoadShaders( vfile.c_str(), ffile.c_str());

	// Get a handle for our "MVP" uniform
	MatrixID = glGetUniformLocation(programID, "MVP");


}




void Skeleton::draw(){
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

void Skeleton::printSkeleton(Node* node){
	if(node==NULL){
		if(root != NULL){
			std::cout << "ROOT AAAA::::" ;
			printSkeleton(root);

		}else{
			return;
		}
	}else{
		if(node->parent !=NULL)
			std::cout << node->parent -> label << " :::: " ;
		std::cout << node->label << " ---------- " << std::endl;
		if(node->child.size()>0){
			for(Node* n : node->child){
				printSkeleton(n);
			}

		}
		else{
			std::cout << "end of bones" << std::endl;
		}
	}
}

void Skeleton::skeletonMesh(Node* node){
	if(node==NULL){
		if(root != NULL){
			skeletonMesh(root);
		}else{
			return;
		}
	}else{
		if(node->parent !=NULL){

			glm::vec4 v= node->parent->ModelMatrix*glm::vec4(glm::vec3(0.0), 1.0);
			std::cout<<glm::to_string(node->parent->ModelMatrix)<<std::endl;
			std::cout<<glm::to_string(node->ModelMatrix)<<std::endl;		
			std::cout << node->parent->label << " " << v.x << " " << v.y << " " << v.z << " -----> ";
			vertices.push_back(glm::vec3(v));
			v= node->ModelMatrix*glm::vec4(glm::vec3(0.0), 1.0);
			std::cout << node->parent->label << " "<< v.x << " " << v.y << " " << v.z << std::endl;
			vertices.push_back(glm::vec3(v));

			colors.push_back(glm::vec3(255.0,0.0,0.0));
			colors.push_back(glm::vec3(255.0,0.0,0.0));

		}
		if(node->child.size()>0){
			for(Node* n : node->child){
				skeletonMesh(n);
			}
		}
		else{
			std::cout << "end of bones" << std::endl;
		}
	}
}


void Skeleton::skeletonVBO(){
	
	glBindVertexArray(VertexArrayID);
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_STATIC_DRAW);
}
