#ifndef PLI_VIS_RENDERABLE_HPP_
#define PLI_VIS_RENDERABLE_HPP_

namespace pli
{
class renderable
{
public:
  virtual ~renderable() = default;

  virtual void initialize() { }
  virtual void render    () { }
};
}

#endif
