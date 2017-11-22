#ifndef ZERNIKE_VERT_GLSL_
#define ZERNIKE_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string zernike_vert = R"(\
#version 450
#extension GL_ARB_explicit_attrib_location : enable

layout(location = 0) in vec3 position;

uniform uvec2 dimensions;
uniform uvec2 spacing   ;

layout(location = 0) out vertex_data 
{
  vec3      relative_position;
  flat uint offset;
} vs_out;

void main()
{
  uvec2 location    = uvec2(gl_InstanceID / dimensions.x % dimensions.y, gl_InstanceID % dimensions.x);
  vec4  translation = vec4(location.x * spacing.x, location.y * spacing.y, 0.0, 1.0);
  vec4  scale       = vec4(             spacing.x,              spacing.y, 1.0, 1.0);

  gl_Position              = vec4(position, 1.0) * scale + translation;
  vs_out.relative_position = position;
  vs_out.offset            = gl_InstanceID;
}
)";
}

#endif