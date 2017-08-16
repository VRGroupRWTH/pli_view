#include <pli_vis/visualization/primitives/point_light.hpp>

namespace pli
{
const float& point_light::range() const
{
  return range_;
}

void point_light::set_range(float range)
{
  range_ = range;
}
}
