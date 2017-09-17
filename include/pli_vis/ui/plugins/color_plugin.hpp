#ifndef PLI_VIS_COLOR_PLUGIN_HPP_
#define PLI_VIS_COLOR_PLUGIN_HPP_

#include <pli_vis/ui/plugin.hpp>
#include <ui_color_toolbox.h>

namespace pli
{
class color_plugin : public plugin<color_plugin, Ui_color_toolbox>
{
public:
  explicit color_plugin(QWidget* parent = nullptr);

  int   mode    () const;
  float k       () const;
  bool  inverted() const;
};
}

#endif
