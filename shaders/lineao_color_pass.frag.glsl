#ifndef LINEAO_COLOR_PASS_FRAG_GLSL_
#define LINEAO_COLOR_PASS_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_color_pass_frag = R"(\
#version 400

in  vec3 vert_direction;
out vec4 frag_color    ;

void main()
{
  frag_color = vec4(abs(vert_direction.x), abs(vert_direction.z), abs(vert_direction.y), 1.0);
}
)";
}

#endif
