#ifndef LINEAO_MAIN_PASS_VERT_GLSL_
#define LINEAO_MAIN_PASS_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_main_pass_vert = R"(\
#version 420

in  vec3 vertex        ;
in  vec2 texcoords     ;
out vec2 vert_texcoords;

void main()
{
  gl_Position    = vec4(vertex, 1.0);
  vert_texcoords = texcoords;
}
)";
}


#endif