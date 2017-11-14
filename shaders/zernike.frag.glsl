#ifndef ZERNIKE_FRAG_GLSL_
#define ZERNIKE_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string zernike_frag = R"(\
#version 450

in vertex_data 
{
  vec3 position;
  uint offset  ;
} fs_in;

layout(std430, binding = 0) readonly buffer Coefficients 
{
  float coefficients[];
};
uniform uint coefficients_per_voxel;

out vec4 color;

void main()
{
  color = vec4(1.0, 0.0, 0.0, 1.0);
}
)";
}

#endif
