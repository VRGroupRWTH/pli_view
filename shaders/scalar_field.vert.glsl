#ifndef SCALAR_FIELD_VERT_GLSL_
#define SCALAR_FIELD_VERT_GLSL_

#include <string>

namespace shaders
{
std::string scalar_field_vert = R"(\
#version 400

uniform mat4 projection    ;
uniform mat4 view          ;
in      vec3 vertex        ;
in      vec2 texcoords     ;
out     vec2 vert_texcoords;

void main()
{
  gl_Position    = projection * view * vec4(vertex, 1.0);
  vert_texcoords = texcoords;
}
)";
}


#endif