#ifndef PLI_VIS_PLUGIN_HPP_
#define PLI_VIS_PLUGIN_HPP_

#include <pli_vis/aspects/loggable.hpp>
#include <pli_vis/ui/plugin_base.hpp>

namespace pli
{
template<typename derived, typename ui_type>
class plugin : public plugin_base, public loggable<derived>, public ui_type
{
public:
  explicit plugin(QWidget* parent = nullptr) : plugin_base(parent) { }
  virtual ~plugin() = default;

  void set_owner(application* owner) override
  {
    owner_ = owner;
  }
  void awake    () override {}
  void start    () override {}
  void destroy  () override {}

protected:
  application* owner_ = nullptr;
};
}

#endif
