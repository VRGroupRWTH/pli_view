#ifndef PLI_VIS_HSL_COLOR_MAPPER_HPP_
#define PLI_VIS_HSL_COLOR_MAPPER_HPP_

#define _USE_MATH_DEFINES

#include <array>
#include <math.h>

#include <convert.hpp>

#include <graphics/color_convert.hpp>

namespace pli
{
class hsl_color_mapper
{
public:
  template<typename color_type, typename vector_type>
  static std::array<color_type, 3> map(const vector_type& vector)
  {
    auto spherical = to_spherical_coords(vector);

    while (spherical[1] <= 0)
      spherical[1] += M_PI;
    while (spherical[2] <= 0)
      spherical[2] += M_PI;
    if (spherical[2] >= M_PI / 2)
      spherical[2] = M_PI - spherical[2];

    return hsl_to_rgb<color_type>(
    {
      spherical[1] /  M_PI     ,
      spherical[2] / (M_PI / 2),
      0.5
    });
  }
};
}

#endif
