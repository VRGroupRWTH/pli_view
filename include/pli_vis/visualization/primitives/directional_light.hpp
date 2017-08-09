#ifndef PLI_VIS_DIRECTIONAL_LIGHT_HPP_
#define PLI_VIS_DIRECTIONAL_LIGHT_HPP_

#include <pli_vis/visualization/primitives/light.hpp>

namespace pli
{
class directional_light : public light
{
public:
  const glm::vec3& direction() const;

  void set_direction(const glm::vec3& direction);

protected:
  glm::vec3 direction_ = glm::vec3(0.0F, 0.0F, 1.0F);
};
}

#endif
