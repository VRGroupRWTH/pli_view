#ifndef PLI_VIS_TRACTOGRAPHY_PLUGIN_HPP_
#define PLI_VIS_TRACTOGRAPHY_PLUGIN_HPP_

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <visualization/linear_tracer.hpp>
#include <ui_tractography_toolbox.h>

namespace pli
{
class tractography_plugin : 
  public plugin, 
  public loggable<tractography_plugin>,
  public Ui_tractography_toolbox
{
public:
  tractography_plugin(QWidget* parent = nullptr);
  void start () override;

private:
  void trace ();

  linear_tracer tracer_;
};
}

#endif
