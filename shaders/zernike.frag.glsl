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

vec2 to_radial(vec2 cartesian)
{
  return vec2(length(cartesian), atan(cartesian.y, cartesian.x));
}

uint factorial(uint n)
{
  uint result = 1;
  for(uint i = 2; i <= n; i++)
    result *= i;
  return result;
}
float mode(uvec2 nm, float rho)
{
  float result = 0.0;
  for(uint i = 0; i < (nm.x - nm.y) / 2; i++)
    result += pow(rho, nm.x - 2 * i) * (pow(-1, i) * factorial(nm.x - i)) / (factorial(i) * factorial((nm.x + nm.y) / 2 - i) * factorial((nm.x - nm.y) / 2 - i));
  return result;
}
float evaluate(uvec2 nm, vec2 rt)
{
  return mode(nm, rt.x) * (nm.y >= 0 ? cos(nm.y * rt.y) : sin(nm.y * rt.y));
}

void main()
{
  uint coefficient_offset = fs_in.offset * coefficients_per_voxel;
  for(uint i = 0; i < coefficients_per_voxel; i++)
  {

  }
  color = vec4(1.0, 1.0, 1.0, 1.0);
}
)";
}
#endif
