#ifndef PLI_VIS_RGB_COLOR_MAPPER_HPP_
#define PLI_VIS_RGB_COLOR_MAPPER_HPP_

#include <array>
#include <math.h>

namespace pli
{
class rgb_color_mapper
{
public:
  template<typename vector_type, typename color_type = vector_type>
  static color_type map(const vector_type& vector)
  {
    return
    {
      abs(vector[0] * 255),
      abs(vector[1] * 255),
      abs(vector[2] * 255)
    };
  }
};
}

#endif
