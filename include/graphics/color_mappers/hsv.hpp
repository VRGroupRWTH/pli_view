#ifndef PLI_VIS_HSV_COLOR_MAPPER_HPP_
#define PLI_VIS_HSV_COLOR_MAPPER_HPP_

#define _USE_MATH_DEFINES

#include <array>
#include <math.h>

#include <cush.h>

#include <graphics/color_convert.hpp>

namespace pli
{
class hsv_color_mapper
{
public:
  template<typename vector_type, typename color_type = vector_type>
  static color_type map(const vector_type& vector)
  {
    auto spherical = to_spherical_coords(vector);

    while (spherical[1] <= 0)
      spherical[1] += M_PI;
    while (spherical[2] <= 0)
      spherical[2] += M_PI;
    if (spherical[2] >= M_PI / 2)
      spherical[2] = M_PI - spherical[2];

    return hsv_to_rgb<color_type>(
    {
      spherical[1] /  M_PI     ,
      spherical[2] / (M_PI / 2),
      1.0
    });
  }
};
}

#endif
