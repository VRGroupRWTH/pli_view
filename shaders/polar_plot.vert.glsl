#ifndef POLAR_PLOT_VERT_GLSL_
#define POLAR_PLOT_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string polar_plot_vert = R"(\
#version 400

uniform  mat4 model         ;
uniform  mat4 view          ;
uniform  mat4 projection    ;
in       vec3 vertex        ;
in       vec3 direction     ;
out      vec3 vert_direction;

void main()
{
  gl_Position    = projection * view * model * vec4(vertex, 1.0);
  vert_direction = direction;
}
)";
}

#endif