#include <pli_vis/visualization/primitives/spot_light.hpp>

namespace pli
{
const float& spot_light::range     () const
{
  return range_;
}
const float& spot_light::spot_angle() const
{
  return spot_angle_;
}

void spot_light::set_range     (float range     )
{
  range_ = range;
}
void spot_light::set_spot_angle(float spot_angle)
{
  spot_angle_ = spot_angle;
}
}
