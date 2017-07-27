#ifndef VIEW_DEPENDENT_VERT_GLSL_
#define VIEW_DEPENDENT_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_vert = R"(\
#version 400

uniform mat4 projection    ;
uniform mat4 view          ;
uniform bool view_dependent;
in      vec3 vertex        ;
in      vec4 color         ;
out     vec4 vert_color    ;

void main()
{
  gl_Position = projection * view * vec4(vertex, 1.0);

  if(view_dependent)
  {
    float alpha = 1.0 - abs(dot(normalize(inverse(projection * view)[2].xyz), normalize(color.xzy)));
    vert_color = vec4(color.xyz, alpha);
  }
  else
  {
    vert_color = color;
  }
}
)";
}


#endif