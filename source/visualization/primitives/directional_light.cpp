#include <pli_vis/visualization/primitives/directional_light.hpp>

namespace pli
{
const glm::vec3& directional_light::direction() const
{
  return direction_;
}

void directional_light::set_direction(const glm::vec3& direction)
{
  direction_ = direction;
}
}
