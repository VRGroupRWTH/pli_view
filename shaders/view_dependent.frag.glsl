#ifndef VIEW_DEPENDENT_FRAG_GLSL_
#define VIEW_DEPENDENT_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_frag = R"(\
#version 400

uniform float     cutoff       = 0.25;
uniform sampler2D depth_texture;
uniform uvec2     screen_size  ;
uniform float     z_near       = 0.1F;
uniform float     z_far        = 10000.0F;
in      vec4      vert_color   ;
out     vec4      frag_color   ;

float get_linearized_depth()
{
  float depth = texture(depth_texture, vec2(gl_FragCoord.x / screen_size.x, gl_FragCoord.y / screen_size.y)).x;
  return (2.0 * z_near) / (z_far + z_near - depth * (z_far - z_near));
}

void main()
{
  if(vert_color.a < cutoff)
  {
    discard;
  }
  else
  {
    float depth = get_linearized_depth();
    frag_color  = vert_color * vec4(depth, depth, depth, 1.0);
  }
}
)";
}

#endif
