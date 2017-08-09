#ifndef PLI_VIS_SPOT_LIGHT_HPP_
#define PLI_VIS_SPOT_LIGHT_HPP_

#include <pli_vis/visualization/primitives/light.hpp>

namespace pli
{
class spot_light : public light
{
public:
  const glm::vec3& position  () const;
  const glm::vec3& direction () const;
  const float&     range     () const;
  const float&     spot_angle() const;

  void set_position  (const glm::vec3& position  );
  void set_direction (const glm::vec3& direction );
  void set_range     (float            range     );
  void set_spot_angle(float            spot_angle);

protected:
  glm::vec3 position_   = glm::vec3(0.0F);
  glm::vec3 direction_  = glm::vec3(0.0F, 0.0F, 1.0F);
  float     range_      = 10.0F;
  float     spot_angle_ = 30.0F;
};
}

#endif
