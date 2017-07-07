#ifndef PLI_VIS_SCALAR_PLUGIN_HPP_
#define PLI_VIS_SCALAR_PLUGIN_HPP_

#include <future>

#include <pli_vis/aspects/loggable.hpp>
#include <pli_vis/ui/plugin.hpp>

#include <ui_scalar_toolbox.h>

namespace pli
{
class scalar_field;

class scalar_plugin : public plugin, public loggable<scalar_plugin>, public Ui_scalar_toolbox
{
public:
  scalar_plugin(QWidget* parent = nullptr);
  void start () override;

private:
  void upload();

  scalar_field*     scalar_field_;
  std::future<void> future_      ;
};
}

#endif
