
#ifndef NODE_H
#define NODE_H


#include <vector>
#include <string>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>




class Node 
{ 
    // Access specifier 
public:

	Node( Node* _parent);
	Node(Node* _parent,std::string _label);
	~Node();

	void addChild(Node* n);
	std::string label;
	std::vector<Node*> child;
	std::vector<glm::vec3> color;

	bool isRoot;
	Node* parent;
	glm::mat4 ModelMatrix;



}; 

#endif