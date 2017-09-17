#ifndef STREAMLINE_RENDERER_FRAG_GLSL_
#define STREAMLINE_RENDERER_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string streamline_renderer_frag = R"(\
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
