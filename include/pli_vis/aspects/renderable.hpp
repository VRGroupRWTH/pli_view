#ifndef PLI_VIS_RENDERABLE_HPP_
#define PLI_VIS_RENDERABLE_HPP_

#include <pli_vis/visualization/primitives/transform.hpp>

namespace pli
{
class camera;

class renderable : public transform
{
public:
  virtual ~renderable() = default;

  virtual void initialize()                     { }
  virtual void render    (const camera* camera) { }

  void set_active(bool active)
  {
    active_ = active;
  }
  bool active    () const
  {
    return active_;
  }

  void set_color_mapping(int mode, float k, bool inverted)
  {
    color_mode_     = mode;
    color_k_        = k;
    color_inverted_ = inverted;
  }

protected:
  bool  active_         = true;
  int   color_mode_     = 0;
  float color_k_        = 0.5f;
  bool  color_inverted_ = false;
};
}

#endif
