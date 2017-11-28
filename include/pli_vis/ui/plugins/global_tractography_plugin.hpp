#ifndef PLI_VIS_GLOBAL_TRACTOGRAPHY_PLUGIN_HPP_
#define PLI_VIS_GLOBAL_TRACTOGRAPHY_PLUGIN_HPP_

#include <pli_vis/ui/plugin.hpp>
#include <ui_global_tractography_toolbox.h>

namespace pli
{
class global_tractography_plugin : public plugin<global_tractography_plugin, Ui_global_tractography_toolbox>
{
public:
  explicit global_tractography_plugin(QWidget* parent = nullptr);

protected:

};
}

#endif