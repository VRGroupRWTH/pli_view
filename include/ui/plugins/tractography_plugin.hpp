#ifndef PLI_VIS_TRACTOGRAPHY_PLUGIN_HPP_
#define PLI_VIS_TRACTOGRAPHY_PLUGIN_HPP_

#include <future>

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_tractography_toolbox.h>
#include <visualization/basic_tracer.hpp>

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

  basic_tracer*     basic_tracer_;
  std::future<void> future_;
};
}

#endif
