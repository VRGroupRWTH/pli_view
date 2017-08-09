#ifndef PLI_VIS_LIGHT_HPP_
#define PLI_VIS_LIGHT_HPP_

#include <glm/glm.hpp>

#include <pli_vis/visualization/primitives/transform.hpp>

namespace pli
{
class light : public transform
{
public:
  const float&     intensity() const;
  const glm::vec3& color    () const;

  void set_intensity(float            intensity);
  void set_color    (const glm::vec3& color    );

protected:
  float     intensity_ = 1.0F;
  glm::vec3 color_     = glm::vec3(1.0F);
};
}

#endif
