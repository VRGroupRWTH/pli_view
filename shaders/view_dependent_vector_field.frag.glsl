#ifndef VIEW_DEPENDENT_FRAG_GLSL_
#define VIEW_DEPENDENT_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_vector_field_frag = R"(\
#version 450

uniform bool  view_dependent = true;
uniform bool  invert         = true;
uniform float rate_of_decay  = 1.0 ;
uniform float cutoff         = 0.25;
uniform mat4  view           ;
out     vec4  frag_color     ;

in vertex_data {
  flat vec3 direction;
} fs_in;
 
void main()
{
  frag_color = vec4(abs(fs_in.direction.x), abs(fs_in.direction.z), abs(fs_in.direction.y), 1.0);
  if(view_dependent)
  {
    float alpha = abs(dot(normalize(inverse(view)[2].xyz), normalize(fs_in.direction)));
    if(invert)
      alpha = 1.0 - alpha;
    if(alpha < cutoff)
      discard;
    frag_color.a = pow(alpha, rate_of_decay);
  }
}
)";
}

#endif
