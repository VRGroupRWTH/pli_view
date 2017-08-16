#ifndef LINEAO_MAIN_PASS_FRAG_GLSL_
#define LINEAO_MAIN_PASS_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_main_pass_frag = R"(\
#version 420

uniform uint      layer_count          = 4  ;
uniform uint      sample_count         = 32 ;
uniform float     radius_multiplier    = 1.5;
uniform sampler2D normal_depth_texture ;
uniform sampler2D color_texture        ;
uniform sampler2D zoom_texture         ;
uniform sampler3D random_texture       ;
in      vec2      vert_texcoords       ;
out     vec4      frag_color           ;

void main()
{ 
  vec4  color        = texture(color_texture       , vert_texcoords);
  vec4  normal_depth = texture(normal_depth_texture, vert_texcoords);
  float zoom         = texture(zoom_texture        , vert_texcoords).x;
  
  float ambient_occlusion    = 0.0;
  uint  current_sample_count = sample_count;
  for(uint i = 0; i < layer_count; i++)
  {
    float current_ambient_occlusion = 0.0;
    for(uint j = 0; j < current_sample_count; j++)
    {
      float visibility = 0.0;
      float weight     = 0.0;
      
      // TODO: Check correctness of the scalar multiplications in the next line.
      vec3 random_vector = zoom * radius_multiplier * i * normalize(texture(random_texture, vec3(vert_texcoords.xy, j)).xyz);
      vec3 random_point  = vec3(vert_texcoords, normal_depth.w) + random_vector;
      
      vec4 occluder_normal_depth = texture(normal_depth_texture, random_point.xy);
      if(normal_depth.w - occluder_normal_depth.w < 0)
        visibility = 1.0;

      // TODO: Adjust weight according to depth and current layer instead of fixed.
      weight = 0.5;

      current_ambient_occlusion += (1.0 - visibility) * weight;
    }
    ambient_occlusion    += current_ambient_occlusion / current_sample_count;
    current_sample_count /= (i + 1);
  }
  ambient_occlusion /= layer_count;

  frag_color = ambient_occlusion * color;

  if(frag_color.a == 0.0)
    discard;
}
)";
}

#endif
