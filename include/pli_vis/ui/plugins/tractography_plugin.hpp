#ifndef PLI_VIS_TRACTOGRAPHY_PLUGIN_HPP_
#define PLI_VIS_TRACTOGRAPHY_PLUGIN_HPP_

#include <future>

#include <pli_vis/ui/plugin.hpp>
#include <pli_vis/visualization/algorithms/streamline_renderer.hpp>
#include <ui_tractography_toolbox.h>

namespace pli
{
class tractography_plugin : public plugin<tractography_plugin, Ui_tractography_toolbox>
{
public:
  explicit tractography_plugin(QWidget* parent = nullptr);

  void start() override;

private:
  void trace();

  std::array<std::size_t, 3> seed_offset() const;
  std::array<std::size_t, 3> seed_size  () const;
  std::array<std::size_t, 3> seed_stride() const;

  streamline_renderer* streamline_renderer_;
  std::future<void>    future_;
};
}

#endif
