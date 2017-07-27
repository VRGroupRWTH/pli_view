#ifndef VIEW_DEPENDENT_VERT_GLSL_
#define VIEW_DEPENDENT_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_vert = R"(\
#version 400

uniform mat4  projection    ;
uniform mat4  view          ;
uniform bool  view_dependent;
uniform float rate_of_decay ;
in      vec3  vertex        ;
in      vec4  color         ;
out     vec4  vert_color    ;

void main()
{
  gl_Position = projection * view * vec4(vertex, 1.0);

  if(view_dependent)
  {
    vert_color = vec4(color.xyz, pow(1.0 - abs(dot(normalize(inverse(view)[2].xyz), normalize(color.xzy))), rate_of_decay));
  }
  else
  {
    vert_color = color;
  }
}
)";
}


#endif