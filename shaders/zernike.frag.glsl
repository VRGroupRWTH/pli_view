#ifndef ZERNIKE_FRAG_GLSL_
#define ZERNIKE_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string zernike_frag = R"(\
#version 450
#extension GL_ARB_explicit_attrib_location : enable

const float pi       = 3.1415926535897932384626433832795;
const float infinity = 1.0 / 0.0;

in vertex_data 
{
  vec3      relative_position;
  flat uint offset;
} fs_in;

layout(std430, binding=0) buffer Coefficients 
{
  float coefficients[];
};
uniform uint coefficients_per_voxel;

layout(location = 0) out vec4 color;

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
float factorial(int n)
{
  float result = 1.0;
  for(int i = 2; i <= n; i++)
    result *= float(i);
  return result;
}
float mode(ivec2 nm, float rho)
{
  float result = 0.0;
  for(int i = 0; i <= (nm.x - nm.y) / 2; i++)
    result += pow(rho, float(nm.x - 2 * i)) * ((mod(i, 2) == 0 ? 1.0 : -1.0) * factorial(nm.x - i)) / (factorial(i) * factorial((nm.x + nm.y) / 2 - i) * factorial((nm.x - nm.y) / 2 - i));
  return result;
}
float evaluate(ivec2 nm, vec2 rt)
{
  return mode(ivec2(abs(nm.x), abs(nm.y)), rt.x) * (nm.y >= 0 ? cos(float(nm.y) * rt.y) : sin(float(-nm.y) * rt.y));
}

void main()
{
  int  coefficient_offset = int(fs_in.offset * coefficients_per_voxel);
  vec2 radial             = to_radial(2.0 * (fs_in.relative_position.xy - vec2(0.5, 0.5)));
  if  (radial.x >= 1.0) discard;
  radial.y += pi;

  float scalar = 0.0;
  for(int i = 0; i < int(coefficients_per_voxel); i++)
    scalar += coefficients[coefficient_offset + i] * evaluate(quantum_index(i), radial);
  
  if(scalar > 0.0) 
    color = vec4(0.0, 0.0, abs(scalar), 1.0);
  else
    color = vec4(abs(scalar), 0.0, 0.0, 1.0);
}
)";
}

#endif
