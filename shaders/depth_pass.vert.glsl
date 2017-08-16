#ifndef DEPTH_PASS_VERT_GLSL_
#define DEPTH_PASS_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string depth_pass_vert = R"(\
#version 400

uniform mat4 model     ;
uniform mat4 view      ;
uniform mat4 projection;
in      vec3 vertex    ;

void main()
{
  gl_Position = projection * view * model * vec4(vertex, 1.0);
}
)";
}


#endif