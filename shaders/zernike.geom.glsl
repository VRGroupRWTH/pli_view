#ifndef ZERNIKE_GEOM_GLSL_
#define ZERNIKE_GEOM_GLSL_

#include <string>

namespace shaders
{
static std::string zernike_geom = R"(\
#version 450

layout (points) in;
layout (triangle_strip, max_vertices = 256) out;

in vertex_data 
{
  vec3  position     ;
  uvec2 sampling_size;
} gs_in[];
 
out vertex_data 
{
  vec2 relative_position;
  uint coefficient_index;
} gs_out;

void main()
{
  // TODO
}
)";
}

#endif
