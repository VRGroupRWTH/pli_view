#ifndef VIEW_DEPENDENT_VERT_GLSL_
#define VIEW_DEPENDENT_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_vector_field_vert = R"(\
#version 450

uniform bool  view_dependent = true;
uniform bool  invert         = true;
uniform float rate_of_decay  = 1.0 ;
uniform mat4  model          ;
uniform mat4  view           ;
in      vec3  direction      ;

out vertex_data {
  vec3 direction;
  vec4 color    ;
} vs_out;

void main()
{
  gl_Position      = vec4(0.0);
  vs_out.direction = direction;
  vs_out.color     = vec4(abs(direction.x), abs(direction.z), abs(direction.y), 1.0);

  if(view_dependent)
  {
    float alpha = abs(dot(normalize(inverse(view)[2].xyz), normalize(inverse(model) * vec4(direction, 1.0)).xyz));
    if(invert)
      alpha = 1.0 - alpha;
    vs_out.color.a = pow(alpha, rate_of_decay);
  }
}
)";
}


#endif