
#include <iostream>

#include "node.h"

#include "common/shader.hpp"
#include "common/texture.hpp"
#include "common/controls.hpp"
#include "common/objloader.hpp"
#include "common/vboindexer.hpp"



Node::Node(Node* _parent){
	parent=_parent;
	isRoot=false;;
	ModelMatrix = glm::mat4(1.0);
}

Node::Node(Node* _parent,std::string _label){
	parent=_parent;
	label=_label;	
	isRoot=false;;
	ModelMatrix = glm::mat4(1.0);
}


Node::~Node(){
	for (Node* n : child){
		delete n;
	}


}


void Node::addChild(Node* n){
	child.push_back(n);
}