#ifndef FULLSCREEN_TEXTURE_FRAG_GLSL_
#define FULLSCREEN_TEXTURE_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string fullscreen_texture_frag = R"(\
#version 400

uniform sampler2D texture_unit  ;
in      vec2      vert_texcoords;
out     vec4      frag_color    ;

void main()
{
  frag_color = texture(texture_unit, vert_texcoords).xyzw;
}
)";
}

#endif
