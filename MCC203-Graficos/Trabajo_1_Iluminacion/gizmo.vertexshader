#version 330 core

// Aqui van los vertex buffer que mandamos al GPU
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexColor;

// datos de salida hacia el fragment shader (lo que tenemos que calcular)
out vec3 vcolor;

// Datos uniformes al objeto
uniform mat4 MVP;

void main(){
	vcolor=vertexColor;
	// gl_position es la position del vertice despues de la proyeccion
	gl_Position =  MVP * vec4(vertexPosition_modelspace,1);
	
	
}

