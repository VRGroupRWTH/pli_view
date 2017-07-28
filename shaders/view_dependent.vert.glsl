#ifndef VIEW_DEPENDENT_VERT_GLSL_
#define VIEW_DEPENDENT_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_vert = R"(\
#version 400

uniform mat4  projection     ;
uniform mat4  view           ;
uniform bool  view_dependent = true;
uniform bool  invert         = true;
uniform float rate_of_decay  = 1.0 ;
in      vec3  vertex         ;
in      vec3  direction      ;
out     vec4  vert_color     ;

void main()
{
  gl_Position = projection * view * vec4(vertex, 1.0);

  vec3 color = vec3(abs(direction.x), abs(direction.z), abs(direction.y));

  if(view_dependent)
  {
    float alpha = abs(dot(normalize(inverse(view)[2].xyz), normalize(direction)));
    if(invert)
      alpha = 1.0 - alpha;
    vert_color = vec4(color, pow(alpha, rate_of_decay));
  }
  else
  {
    vert_color = vec4(color, 1.0);
  }
}
)";
}


#endif