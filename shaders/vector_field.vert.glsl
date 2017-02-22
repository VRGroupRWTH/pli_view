#ifndef VECTOR_FIELD_VERT_GLSL_
#define VECTOR_FIELD_VERT_GLSL_

#include <string>

namespace shaders
{
std::string vector_field_vert = R"(\
#version 400

in  vec3 vertex    ;
in  vec4 color     ;
out vec4 vert_color;

void main()
{
  gl_Position = vec4(vertex, 1.0);
  vert_color  = color;
}
)";
}


#endif