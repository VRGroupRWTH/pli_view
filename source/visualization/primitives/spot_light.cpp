#include <pli_vis/visualization/primitives/spot_light.hpp>

namespace pli
{
const glm::vec3& spot_light::position  () const
{
  return position_;
}
const glm::vec3& spot_light::direction () const
{
  return direction_;
}
const float&     spot_light::range     () const
{
  return range_;
}
const float&     spot_light::spot_angle() const
{
  return spot_angle_;
}

void spot_light::set_position  (const glm::vec3& position  )
{
  position_ = position;
}
void spot_light::set_direction (const glm::vec3& direction )
{
  direction_ = direction;
}
void spot_light::set_range     (float            range     )
{
  range_ = range;
}
void spot_light::set_spot_angle(float            spot_angle)
{
  spot_angle_ = spot_angle;
}
}
