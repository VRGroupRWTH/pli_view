#ifndef SCALAR_FIELD_FRAG_GLSL_
#define SCALAR_FIELD_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string scalar_field_frag = R"(\
#version 400

uniform sampler2D texture_unit  ;
in      vec2      vert_texcoords;
out     vec4      frag_color    ;

void main()
{
  float value = texture(texture_unit, vert_texcoords).x;
  frag_color  = vec4(value, value, value, value);
}
)";
}

#endif
