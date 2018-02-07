#ifndef VIEW_DEPENDENT_GEOM_GLSL_
#define VIEW_DEPENDENT_GEOM_GLSL_

#include <string>

namespace shaders
{
static std::string view_dependent_vector_field_geom = R"(\
#version 450

layout (points) in;
layout (line_strip, max_vertices = 2) out;

uniform uint  vectors_per_point = 1;
uniform float scale             = 1.0;
uniform uvec3 dimensions        ;
uniform mat4  model             ;
uniform mat4  view              ;
uniform mat4  projection        ;

in  vertex_data {
  vec3 direction;
} gs_in[];
 
out vertex_data {
  flat vec3 direction;
} gs_out;

void main()
{
  mat4 mvp = projection * view * model;

  uvec3 position;
  uint index = gl_PrimitiveIDIn;
  uint t     = index % vectors_per_point;
  uint z     = ((index - t) / vectors_per_point) % dimensions.z;
  uint y     = ((index - z * vectors_per_point - t) / (vectors_per_point * dimensions.z)) % dimensions.y;
  uint x     = ((index - y * dimensions.z * vectors_per_point - z * vectors_per_point - t) / (vectors_per_point * dimensions.z * dimensions.y));

  vec4 direction   = vec4(gs_in[0].direction * 0.5 * scale, 0.0);
  gs_out.direction =  gs_in[0].direction; gl_Position = mvp * (vec4(x, y, z, 1.0) + direction); EmitVertex();
  gs_out.direction = -gs_in[0].direction; gl_Position = mvp * (vec4(x, y, z, 1.0) - direction); EmitVertex();
  EndPrimitive();
}
)";
}

#endif
