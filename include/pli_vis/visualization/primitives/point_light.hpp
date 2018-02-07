#ifndef PLI_VIS_POINT_LIGHT_HPP_
#define PLI_VIS_POINT_LIGHT_HPP_

#include <pli_vis/visualization/primitives/light.hpp>

namespace pli
{
class point_light : public light
{
public:
  const float& range() const;

  void set_range(float range);

protected:
  float range_ = 10.0F;
};
}

#endif
