#ifndef PLI_VIS_VECTOR_RGB_MAPPER_HPP_
#define PLI_VIS_VECTOR_RGB_MAPPER_HPP_

#include <array>

namespace pli
{
class vector_rgb_mapper
{
public:
  template<typename color_type, typename vector_type>
  static std::array<color_type, 3> map(const vector_type& vector)
  {
    return
    {
      color_type(abs(vector[0] * 255)),
      color_type(abs(vector[1] * 255)),
      color_type(abs(vector[2] * 255))
    };
  }
};
}

#endif
