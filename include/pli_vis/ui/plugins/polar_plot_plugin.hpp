#ifndef PLI_VIS_POLAR_PLOT_PLUGIN_HPP_
#define PLI_VIS_POLAR_PLOT_PLUGIN_HPP_

#include <future>

#include <pli_vis/ui/plugin.hpp>
#include <pli_vis/visualization/algorithms/polar_plot_field.hpp>
#include <ui_polar_plot_toolbox.h>

namespace pli
{
class polar_plot_plugin : public plugin<polar_plot_plugin, Ui_polar_plot_toolbox>
{
public:
  explicit polar_plot_plugin(QWidget* parent = nullptr);

  void start() override;

private:
  std::future<void> future_;
  polar_plot_field* field_ = nullptr;
};
}

#endif
