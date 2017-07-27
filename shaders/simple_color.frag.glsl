#ifndef SIMPLE_COLOR_FRAG_GLSL_
#define SIMPLE_COLOR_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string simple_color_frag = R"(\
#version 400

in  vec4 vert_color;
out vec4 frag_color;

void main()
{
  frag_color = vert_color;
}
)";
}

#endif
