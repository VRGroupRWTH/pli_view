#ifndef PLI_VIS_INTERACTOR_PLUGIN_HPP_
#define PLI_VIS_INTERACTOR_PLUGIN_HPP_

#include <pli_vis/aspects/loggable.hpp>
#include <pli_vis/ui/plugin.hpp>

#include <ui_interactor_toolbox.h>

namespace pli
{
class interactor_plugin : public plugin, public loggable<interactor_plugin>, public Ui_interactor_toolbox
{
public:
  interactor_plugin(QWidget* parent = nullptr);
  void start () override;
};
}

#endif
