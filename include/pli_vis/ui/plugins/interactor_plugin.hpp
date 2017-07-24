#ifndef PLI_VIS_INTERACTOR_PLUGIN_HPP_
#define PLI_VIS_INTERACTOR_PLUGIN_HPP_

#include <pli_vis/ui/plugin.hpp>

#include <ui_interactor_toolbox.h>

namespace pli
{
class interactor_plugin : public plugin<interactor_plugin, Ui_interactor_toolbox>
{
public:
  explicit interactor_plugin(QWidget* parent = nullptr);

  void start() override;
};
}

#endif
