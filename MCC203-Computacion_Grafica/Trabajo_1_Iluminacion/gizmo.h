
#ifndef GIZMO_H
#define GIZMO_H


#include <vector>
#include <string>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>




class Gizmo 
{ 
    // Access specifier 
public:

 Gizmo();
 ~Gizmo();

 void setShaders(std::string vfile,std::string ffile);
 void loadMesh();
 void draw();

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



}; 

#endif