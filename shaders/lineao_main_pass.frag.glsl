#ifndef LINEAO_MAIN_PASS_FRAG_GLSL_
#define LINEAO_MAIN_PASS_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_main_pass_frag = R"(\
#version 400

uniform sampler2D normal_depth_texture;
uniform sampler2D color_texture       ;
uniform sampler2D zoom_texture        ;
uniform sampler3D random_texture      ;
in      vec2      vert_texcoords      ;
out     vec4      frag_color          ;

void main()
{
  frag_color = texture(random_texture, vec3(vert_texcoords, 0.0));

  if(frag_color.a == 0.0)
    discard;
}
)";
}

#endif
