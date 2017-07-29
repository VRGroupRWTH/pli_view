#ifndef VIEW_DEPENDENT_GEOM_GLSL_
#define VIEW_DEPENDENT_GEOM_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_vector_field_geom = R"(\
#version 450

layout (points) in;
layout (line_strip, max_vertices = 2) out;

uniform float scale      = 1.0;
uniform uvec3 dimensions ;
uniform mat4  projection ;
uniform mat4  view       ;

in  vertex_data {
  vec3 direction;
  vec4 color    ;
} gs_in[];
 
out vertex_data {
  vec3 direction;
  vec4 color    ;
} gs_out;

void main()
{
  uvec3 position;
  uint index = gl_PrimitiveIDIn;
  uint x     = index / (dimensions.y * dimensions.z);
  index     -= x * dimensions.y * dimensions.z;
  uint y     = index / dimensions.z;
  uint z     = index % dimensions.z;

  vec4 direction   = vec4(gs_in[0].direction * 0.5 * scale, 0.0);
  gs_out.direction = gs_in[0].direction;
  gs_out.color     = gs_in[0].color    ;
  gl_Position      = projection * view * (vec4(x, y, z, 1.0) + direction); EmitVertex();
  gl_Position      = projection * view * (vec4(x, y, z, 1.0) - direction); EmitVertex();
  EndPrimitive();
}
)";
}

#endif
