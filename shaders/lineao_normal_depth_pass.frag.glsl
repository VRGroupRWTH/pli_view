#ifndef LINEAO_NORMAL_DEPTH_PASS_FRAG_GLSL_
#define LINEAO_NORMAL_DEPTH_PASS_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_normal_depth_pass_frag = R"(\
#version 400

out vec4 color;

void main()
{
  color = vec4(1.0, 0.0, 0.0, 1.0);
}
)";
}

#endif
