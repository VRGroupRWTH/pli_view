#ifndef COLOR_MAPPER_HPP_
#define COLOR_MAPPER_HPP_

#include <math.h>

#include <vector_types.h>

#include <pli_vis/cuda/utility/convert.h>
#include <pli_vis/cuda/utility/vector_ops.h>

namespace color_mapper
{
namespace detail
{
template<typename type>
type   convert    (const type& input)
{
  auto output = pli::to_spherical_coords(input);

  if (output.y < 0.0F)
    output.y += M_PI;
  if (output.y >= M_PI)
    output.y -= M_PI;
  output.y = M_PI - output.y;

  if (output.z < 0.0F)
    output.z = abs(output.z);
  if (output.z >= M_PI / 2.0F) 
    output.z = M_PI - output.z;

  return output;
}
template<typename type>
float4 hue_to_rgba(const type& hue  )
{
  return clamp(float4
  {
    abs(hue * 6.0F - 3.0F) - 1.0F, 
    2.0F - abs(hue * 6.0F - 2.0F), 
    2.0F - abs(hue * 6.0F - 4.0F), 
    1.0F
  }, 0.0F, 1.0F);
}
template<typename type>
float4 hsl_to_rgba(const type& hsl  )
{
  return (detail::hue_to_rgba(hsl.x) - 0.5F) * (1.0F - abs(2.0F * hsl.z - 1.0F)) * hsl.y + hsl.z;
}
template<typename type>
float4 hsv_to_rgba(const type& hsv  )
{
  return ((detail::hue_to_rgba(hsv.x) - 1.0F) * hsv.y + 1.0F) * hsv.z;
}
}

template<typename type>
float4 to_hsl (const type& vector, const bool fixed_saturation = true, const float fixed_value = 0.5F)
{
  auto converted = detail::convert(vector);
  auto theta     = converted.y / float(M_PI);
  auto phi       = converted.z / float(M_PI / 2.0F);
  return fixed_saturation 
    ? detail::hsl_to_rgba(float3{theta, fixed_value, phi}) 
    : detail::hsl_to_rgba(float3{theta, phi, fixed_value});
}
template<typename type>
float4 to_hsv (const type& vector, const bool fixed_saturation = true, const float fixed_value = 0.5F)
{
  auto converted = detail::convert(vector);
  auto theta     = converted.y / float(M_PI);
  auto phi       = converted.z / float(M_PI / 2.0F);
  return fixed_saturation 
    ? detail::hsv_to_rgba(float3{theta, fixed_value, phi}) 
    : detail::hsv_to_rgba(float3{theta, phi, fixed_value});
}
template<typename type>
float4 to_rgb (const type& vector)
{
  return float4 {abs(vector.x), abs(vector.z), abs(vector.y), 1.0};
}
}

#endif
