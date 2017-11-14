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

ivec2 quantum_index(int index)
{
  ivec2 nm;
  nm.x = int(ceil((-3.0 + sqrt(float(9 + 8 * index))) / 2.0));
  nm.y = 2 * index - nm.x * (nm.x + 2);
  return nm;
}
uint factorial(uint n)
{
  uint result = 1;
  for(uint i = 2; i <= n; i++)
    result *= i;
  return result;
}
float mode(ivec2 nm, float rho)
{
  float result = 0.0;
  for(uint i = 0; i < (nm.x - nm.y) / 2; i++)
    result += pow(rho, nm.x - 2 * i) * (pow(-1, i) * factorial(nm.x - i)) / (factorial(i) * factorial((nm.x + nm.y) / 2 - i) * factorial((nm.x - nm.y) / 2 - i));
  return result;
}
float evaluate(ivec2 nm, vec2 rt)
{
  return mode(nm, rt.x) * (nm.y >= 0 ? cos(nm.y * rt.y) : sin(nm.y * rt.y));
}

void main()
{
  uint coefficient_offset = fs_in.offset * coefficients_per_voxel;

  float scalar = 0.0f;
  for(uint i = 0; i < coefficients_per_voxel; i++)
    scalar += coefficients[coefficient_offset + i] * evaluate(quantum_index(i), to_radial(fs_in.position.xy - vec2(0.5, 0.5)));

  color = vec4(scalar, scalar, scalar, 1.0);
}
)";
}

#endif
