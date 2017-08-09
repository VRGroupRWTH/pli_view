#ifndef PLI_VIS_SPOT_LIGHT_HPP_
#define PLI_VIS_SPOT_LIGHT_HPP_

#include <pli_vis/visualization/primitives/light.hpp>

namespace pli
{
class spot_light : public light
{
public:
  const float& range     () const;
  const float& spot_angle() const;

  void set_range     (float range     );
  void set_spot_angle(float spot_angle);

protected:
  float range_      = 10.0F;
  float spot_angle_ = 30.0F;
};
}

#endif
