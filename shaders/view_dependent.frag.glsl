#ifndef VIEW_DEPENDENT_FRAG_GLSL_
#define VIEW_DEPENDENT_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_frag = R"(\
#version 400

uniform bool  view_dependent = true;
uniform bool  invert         = true;
uniform float rate_of_decay  = 1.0 ;
uniform float cutoff         = 0.25;
uniform mat4  view           ;
in      vec3  vert_direction ;
out     vec4  frag_color     ;

void main()
{
  frag_color = vec4(abs(vert_direction.x), abs(vert_direction.z), abs(vert_direction.y), 1.0);
  if(view_dependent)
  {
    float alpha = abs(dot(normalize(inverse(view)[2].xyz), normalize(vert_direction)));
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
