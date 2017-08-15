#ifndef LINEAO_MAIN_PASS_FRAG_GLSL_
#define LINEAO_MAIN_PASS_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_main_pass_frag = R"(\
#version 420

uniform sampler2D normal_depth_texture;
uniform sampler2D color_texture       ;
uniform sampler2D zoom_texture        ;
uniform sampler3D random_texture      ;
uniform uint      sample_count        ;
in      vec2      vert_texcoords      ;
out     vec4      frag_color          ;

void main()
{
  const float sample_count_reciprocal = 1.0 / float(sample_count);
  const float falloff                 = 0.00001;

  vec4  normal_depth = texture(normal_depth_texture, vert_texcoords);
  vec4  color        = texture(color_texture       , vert_texcoords);
  float zoom         = texture(zoom_texture        , vert_texcoords).x;

  frag_color = normal_depth * color * zoom;

  if(frag_color.a == 0.0)
    discard;
}
)";
}

#endif
