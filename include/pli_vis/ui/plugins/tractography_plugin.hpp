#ifndef PLI_VIS_TRACTOGRAPHY_PLUGIN_HPP_
#define PLI_VIS_TRACTOGRAPHY_PLUGIN_HPP_

#include <future>

#include <pli_vis/aspects/loggable.hpp>
#include <pli_vis/ui/plugin.hpp>
#include <pli_vis/visualization/streamline_tracer.hpp>

#include <ui_tractography_toolbox.h>

namespace pli
{
class tractography_plugin : public plugin, public loggable<tractography_plugin>, public Ui_tractography_toolbox
{
public:
  tractography_plugin(QWidget* parent = nullptr);
  void start () override;

private:
  void trace ();

  streamline_tracer* streamline_tracer_;
  std::future<void>  future_;
};
}

#endif
