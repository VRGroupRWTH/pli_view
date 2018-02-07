#ifndef PLI_VIS_SCALAR_PLUGIN_HPP_
#define PLI_VIS_SCALAR_PLUGIN_HPP_

#include <pli_vis/ui/plugin.hpp>
#include <ui_scalar_toolbox.h>

namespace pli
{
class scalar_field;

class scalar_plugin : public plugin<scalar_plugin, Ui_scalar_toolbox>
{
public:
  explicit scalar_plugin(QWidget* parent = nullptr);

  void start() override;

private:
  void upload();

  scalar_field* scalar_field_;
};
}

#endif
