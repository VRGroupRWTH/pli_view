#ifndef SIMPLE_COLOR_TEXTURE_FRAG_GLSL_
#define SIMPLE_COLOR_TEXTURE_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string simple_color_texture_frag = R"(\
#version 400

uniform sampler2D texture_unit  ;
in      vec2      vert_texcoords;
out     vec4      frag_color    ;

void main()
{
  vec4 color = texture(texture_unit, vert_texcoords);
  if(color.a == 0.0f) discard;
  frag_color = color;
}
)";
}

#endif
