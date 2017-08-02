#ifndef SIMPLE_COLOR_VERT_GLSL_
#define SIMPLE_COLOR_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string simple_color_vert = R"(\
#version 400

uniform mat4 model     ;
uniform mat4 view      ;
uniform mat4 projection;
in      vec3 vertex    ;
in      vec4 color     ;
out     vec4 vert_color;

void main()
{
  gl_Position = projection * view * model * vec4(vertex, 1.0);
  vert_color  = color;
}
)";
}


#endif