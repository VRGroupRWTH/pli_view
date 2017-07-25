#ifndef PLI_VIS_FOM_PLUGIN_HPP_
#define PLI_VIS_FOM_PLUGIN_HPP_

#include <pli_vis/ui/plugin.hpp>
#include <ui_fom_toolbox.h>

namespace pli
{
class vector_field;

class fom_plugin : public plugin<fom_plugin, Ui_fom_toolbox>
{
public:
  explicit fom_plugin(QWidget* parent = nullptr);

  void start() override;

private:
  void upload();

  vector_field* vector_field_;
};
}

#endif
