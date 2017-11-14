#ifndef ZERNIKE_VERT_GLSL_
#define ZERNIKE_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string zernike_vert = R"(\
#version 450
#extension GL_ARB_explicit_attrib_location : enable

layout(location = 0) in vec3 position;

uniform mat4  model     ;
uniform mat4  view      ;
uniform mat4  projection;
uniform uvec2 dimensions;
uniform uvec2 spacing   ;

out vertex_data 
{
  vec3 position;
  uint offset  ;
} vs_out;

void main()
{
  gl_Position     = projection * view * model * vec4(position, 1.0);
  vs_out.position = position;
  vs_out.offset   = 0;
}
)";
}


#endif