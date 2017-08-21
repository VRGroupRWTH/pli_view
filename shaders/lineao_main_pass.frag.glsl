#ifndef LINEAO_MAIN_PASS_FRAG_GLSL_
#define LINEAO_MAIN_PASS_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_main_pass_frag = R"(\
#version 420

uniform uint      layer_count          = 4      ;
uniform uint      sample_count         = 32     ;
uniform float     radius_screen_space  = 16     ;
uniform float     radius_0             = 1.5    ;
uniform float     falloff_0            = 0.00001;
uniform vec2      screen_size          ;
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
  
  float radius       = (zoom * radius_screen_space / screen_size.x) / (1.0 - normal_depth.w);
  float radius_scale = 0.0F;

  float ambient_occlusion    = 0.0;
  uint  current_sample_count = sample_count;
  for(uint i = 0; i < layer_count; i++)
  {
    radius_scale += radius_0 + i;
    float current_ambient_occlusion = 0.0;
    for(uint j = 0; j < current_sample_count; j++)
    {
      float visibility = 0.0;
      float weight     = 0.0;
      
      vec3 random_vector = radius * radius_scale * normalize(texture(random_texture, vec3(vert_texcoords.xy, j)).xyz);
      vec3 random_point  = vec3(vert_texcoords, normal_depth.w) + random_vector;
      
      vec4  occluder_normal_depth = texture(normal_depth_texture, random_point.xy);
      float delta_depth           = normal_depth.w - occluder_normal_depth.w;
      if(delta_depth < 0.0)
        visibility = 1.0;

      float falloff_layer = pow(1.0 - float(i) / float(layer_count), 2);
      if (delta_depth > falloff_layer)
        weight = 0.0;
      else if (delta_depth < falloff_0)
        weight = 1.0;
      else
      {
        float x = (occluder_normal_depth.w - falloff_0) / (falloff_layer - falloff_0);
        weight  = 1.0 - 3.0 * pow(x, 2) + 2.0 * pow(x, 3);
      }

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
