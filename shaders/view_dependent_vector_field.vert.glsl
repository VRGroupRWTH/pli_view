#ifndef VIEW_DEPENDENT_VERT_GLSL_
#define VIEW_DEPENDENT_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_vector_field_vert = R"(\
#version 450

in vec3 direction;

out vertex_data 
{
  vec3 direction;
} vs_out;

void main()
{
  gl_Position      = vec4(0.0);
  vs_out.direction = direction;
}
)";
}


#endif