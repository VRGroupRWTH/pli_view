#include <pli_vis/visualization/primitives/light.hpp>

namespace pli
{
const float&     light::intensity() const
{
  return intensity_;
}
const glm::vec3& light::color    () const
{
  return color_;
}

void light::set_intensity(float            intensity)
{
  intensity_ = intensity;
}
void light::set_color    (const glm::vec3& color    )
{
  color_ = color;
}
}
