#ifndef LINEAO_COLOR_PASS_FRAG_GLSL_
#define LINEAO_COLOR_PASS_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_color_pass_frag = R"(\
#version 400

out vec4 color;

void main()
{
  color = vec4(0.0, 1.0, 0.0, 1.0);
}
)";
}

#endif
