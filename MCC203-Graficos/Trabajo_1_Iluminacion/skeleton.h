
#ifndef SKELETON_H
#define SKELETON_H


#include <vector>
#include <string>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>


class Node;

class Skeleton 
{ 
    // Access specifier 
public:

	Skeleton();
	~Skeleton();

	void setShaders(std::string vfile,std::string ffile);
	void loadMesh();
	void draw();
	void printSkeleton(Node* n=NULL);
	void skeletonMesh(Node* n=NULL);
	void skeletonVBO();

	std::string vertShaderFilename;
	std::string fragShaderFilename;


	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> colors;

	GLuint programID;
	GLuint MatrixID;

	GLuint ModelMatrixID;
	GLuint VertexArrayID;

	GLuint vertexbuffer;
	GLuint colorbuffer;

	glm::mat4 ModelMatrix;

	Node* root;


}; 

#endif