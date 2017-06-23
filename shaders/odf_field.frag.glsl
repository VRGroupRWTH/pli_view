#ifndef ODF_FIELD_FRAG_GLSL_
#define ODF_FIELD_FRAG_GLSL_

#include <string>

namespace shaders
{
std::string odf_field_frag = R"(\
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
