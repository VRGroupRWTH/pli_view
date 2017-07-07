#ifndef PLI_VIS_INTERACTOR_PLUGIN_HPP_
#define PLI_VIS_INTERACTOR_PLUGIN_HPP_

#include <aspects/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_interactor_toolbox.h>

namespace pli
{
class interactor_plugin : 
  public plugin, 
  public loggable<interactor_plugin>, 
  public Ui_interactor_toolbox
{
public:
  interactor_plugin(QWidget* parent = nullptr);
  void start () override;
};
}

#endif
