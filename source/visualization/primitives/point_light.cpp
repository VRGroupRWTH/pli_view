#include <pli_vis/visualization/primitives/point_light.hpp>

namespace pli
{
const glm::vec3& point_light::position() const
{
  return position_;
}
const float&     point_light::range   () const
{
  return range_;
}

void point_light::set_position(const glm::vec3& position)
{
  position_ = position;
}
void point_light::set_range   (float            range   )
{
  range_ = range;
}
}
