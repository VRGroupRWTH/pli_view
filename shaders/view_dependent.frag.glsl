#ifndef VIEW_DEPENDENT_FRAG_GLSL_
#define VIEW_DEPENDENT_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_frag = R"(\
#version 450

uniform float cutoff     = 0.25;
out     vec4  frag_color ;

in vertex_data {
  vec3 direction;
  vec4 color    ;
} fs_in;
 
void main()
{
  if(fs_in.color.a < cutoff)
  {
    discard;
  }
  else
  {
    frag_color = fs_in.color;
  }
}
)";
}

#endif
