#ifndef PLI_VIS_POINT_LIGHT_HPP_
#define PLI_VIS_POINT_LIGHT_HPP_

#include <pli_vis/visualization/primitives/light.hpp>

namespace pli
{
class point_light : public light
{
public:
  const glm::vec3& position() const;
  const float&     range   () const;

  void set_position(const glm::vec3& position);
  void set_range   (float            range   );

protected:
  glm::vec3 position_ = glm::vec3(0.0F);
  float     range_    = 10.0F;
};
}

#endif
