#ifndef PLI_VIS_GLOBAL_TRACTOGRAPHY_PLUGIN_HPP_
#define PLI_VIS_GLOBAL_TRACTOGRAPHY_PLUGIN_HPP_

#include <future>

#include <pli_vis/ui/plugin.hpp>
#include <pli_vis/visualization/algorithms/streamline_renderer.hpp>
#include <ui_global_tractography_toolbox.h>

namespace pli
{
class global_tractography_plugin : public plugin<global_tractography_plugin, Ui_global_tractography_toolbox>
{
public:
  explicit global_tractography_plugin(QWidget* parent = nullptr);

  void start() override;
  
protected:
  void trace();

  std::future<void>    future_;
  streamline_renderer* streamline_renderer_ = nullptr;
};
}

#endif