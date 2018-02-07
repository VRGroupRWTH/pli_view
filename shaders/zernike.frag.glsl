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
uniform uint  coefficients_per_voxel;
uniform int   color_mode     = 0    ;
uniform float color_k        = 0.5  ;
uniform bool  color_inverted = false;

layout(location = 0) out vec4 color;

vec3 hue_to_rgb(float hue)
{
  float R = abs(hue * 6 - 3) - 1;
  float G = 2 - abs(hue * 6 - 2);
  float B = 2 - abs(hue * 6 - 4);
  return clamp(vec3(R,G,B), 0, 1);
}
vec3 hsv_to_rgb(vec3 hsv)
{
  vec3 rgb = hue_to_rgb(hsv.x);
  return ((rgb - 1.0) * hsv.y + 1.0) * hsv.z;
}
vec3 hsl_to_rgb(vec3 hsl)
{
  vec3 rgb = hue_to_rgb(hsl.x);
  float C = (1 - abs(2 * hsl.z - 1)) * hsl.y;
  return (rgb - 0.5) * C + hsl.z;
}
vec3 map_color(vec2 radial, float scalar)
{
  if(radial.y <  0.0)            radial.y += radians(180.0);
  if(radial.y >= radians(180.0)) radial.y -= radians(180.0);
  radial.y = radians(180.0) - radial.y;

  float t = radial.y / radians(180.0);
  if(color_inverted)
    scalar = 1.0 - scalar;

  if(color_mode == 0)
    return hsl_to_rgb(vec3(t, color_k, scalar));
  if(color_mode == 1)
    return hsl_to_rgb(vec3(t, scalar, color_k));
  if(color_mode == 2)
    return hsv_to_rgb(vec3(t, color_k, scalar));
  if(color_mode == 3)
    return hsv_to_rgb(vec3(t, scalar, color_k));
  return vec3(scalar, scalar, scalar);
}

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
  return (nm.y >= 0 ? 1.0f : -1.0f) * sqrt((2.0f * float(nm.x) + 2.0f) / (1.0f + (nm.y == 0 ? 1.0f : 0.0f))) * mode(ivec2(abs(nm.x), abs(nm.y)), rt.x) * (nm.y >= 0 ? cos(float(nm.y) * rt.y) : sin(float(-nm.y) * rt.y));
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
  
  color = vec4(map_color(radial, abs(scalar)), 1.0);
}
)";
}

#endif
