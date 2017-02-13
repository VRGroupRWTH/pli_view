#ifndef PLI_IO_CONVERT_HPP_
#define PLI_IO_CONVERT_HPP_

#include <cmath>
 
#include <decorators.h>

namespace pli
{
template<typename input_type, typename output_type = input_type>
COMMON output_type to_spherical_coords(const input_type& input)
{
  output_type output;
  output[0] = std::sqrt (std::pow(input[0], 2) + std::pow(input[1], 2) + std::pow(input[2], 2));
  output[1] = std::atan2(input[1] , input [0]);
  output[2] = std::acos (input[2] / output[0]);
  return output;
}
template<typename input_type, typename output_type = input_type>
COMMON output_type to_cartesian_coords(const input_type& input)
{
  output_type output;
  output[0] = input[0] * std::cos(input[1]) * std::sin(input[2]);
  output[1] = input[0] * std::sin(input[1]) * std::sin(input[2]);
  output[2] = input[0] * std::cos(input[2]);
  return output;
}

template<typename input_type, typename output_type = input_type>
COMMON output_type to_spherical_coords_2(const input_type& input)
{
  output_type output;
  output.x = std::sqrt (std::pow(input.x, 2) + std::pow(input.y, 2) + std::pow(input.z, 2));
  output.y = std::atan2(input.y , input .x);
  output.z = std::acos (input.z / output.x);
  return output;
}
template<typename input_type, typename output_type = input_type>
COMMON output_type to_cartesian_coords_2(const input_type& input)
{
  output_type output;
  output.x = input.x * std::cos(input.y) * std::sin(input.z);
  output.y = input.x * std::sin(input.y) * std::sin(input.z);
  output.z = input.x * std::cos(input.z);
  return output;
}
}

#endif
