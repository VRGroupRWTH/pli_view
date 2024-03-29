#ifndef LINEAO_ZOOM_PASS_VERT_GLSL_
#define LINEAO_ZOOM_PASS_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_zoom_pass_vert = R"(\
#version 400

uniform  mat4  model     ;
uniform  mat4  view      ;
uniform  mat4  projection;
in       vec3  vertex    ;
flat out float vert_zoom ;

void main()
{
  vec4 position = projection * view * model * vec4(vertex, 1.0);
  gl_Position   = position;
  vert_zoom     = length(position);
}
)";
}


#endif