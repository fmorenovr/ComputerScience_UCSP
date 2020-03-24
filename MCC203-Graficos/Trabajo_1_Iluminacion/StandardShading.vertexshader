#version 330 core

// Aqui van los vertex buffer que mandamos al GPU
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 vertexUV;
layout(location = 2) in vec3 vertexNormal_modelspace;

// datos de salida hacia el fragment shader (lo que tenemos que calcular)
out vec2 UV;
out vec3 Position_worldspace;
out vec3 Normal_cameraspace;
out vec3 EyeDirection_cameraspace;
out vec3 LightDirection_cameraspace;

// Datos uniformes al objeto
uniform mat4 MVP;
uniform mat4 V;
uniform mat4 M;
uniform vec3 LightPosition_worldspace;

void main(){

	// gl_position es la position del vertice despues de la proyeccion
	gl_Position =  MVP * vec4(vertexPosition_modelspace,1);
	
	// La posicion del vertice solamente despues de la transformacion espacial (rotacion)
	Position_worldspace = (M * vec4(vertexPosition_modelspace,1)).xyz;
	
	// Vector que va del vertice hacia la camara
	// En el espacio camara la posicion de la camara es 0,0,0
	vec3 vertexPosition_cameraspace = ( V * M * vec4(vertexPosition_modelspace,1)).xyz;
	EyeDirection_cameraspace = vec3(0,0,0) - vertexPosition_cameraspace;

	// Vector que va del vertice hacia la camara( espacio camara) No hay M porque no transformamos la luz.
	vec3 LightPosition_cameraspace = ( V * vec4(LightPosition_worldspace,1)).xyz;
	LightDirection_cameraspace = LightPosition_cameraspace - vertexPosition_cameraspace;
	
	// Normal despues de la transformacion 
	Normal_cameraspace = ( transpose(inverse(V * M)) * vec4(vertexNormal_modelspace,0)).xyz;
	
	// UV no hacemos nada mas que interpolacion
	UV = vertexUV;
}

