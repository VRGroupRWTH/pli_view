#ifndef VIEW_DEPENDENT_FRAG_GLSL_
#define VIEW_DEPENDENT_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_frag = R"(\
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
