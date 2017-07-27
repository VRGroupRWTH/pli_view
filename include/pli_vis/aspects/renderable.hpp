#ifndef PLI_VIS_RENDERABLE_HPP_
#define PLI_VIS_RENDERABLE_HPP_

namespace pli
{
class camera;

class renderable
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

protected:
  bool active_ = true;
};
}

#endif
