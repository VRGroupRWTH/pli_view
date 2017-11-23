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
  uvec2 size        = dimensions * spacing;
  uvec2 location    = uvec2(gl_InstanceID / dimensions.x % dimensions.y, gl_InstanceID % dimensions.x);
  vec3  translation = vec3(location.x * spacing.x, location.y * spacing.y, 0.0);
  vec3  scale       = vec3(             spacing.x,              spacing.y, 1.0);

  gl_Position              = vec4(position * scale + translation, 1.0) / vec4(size.x, size.y, 1.0, 1.0);
  vs_out.relative_position = position;
  vs_out.offset            = gl_InstanceID;
}
)";
}

#endif