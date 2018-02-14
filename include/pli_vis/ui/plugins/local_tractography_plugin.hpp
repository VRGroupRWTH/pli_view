#ifndef PLI_VIS_LOCAL_TRACTOGRAPHY_PLUGIN_HPP_
#define PLI_VIS_LOCAL_TRACTOGRAPHY_PLUGIN_HPP_

#include <future>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/ui/plugin.hpp>
#include <ui_local_tractography_toolbox.h>

namespace pli
{
class local_tractography_plugin : public plugin<local_tractography_plugin, Ui_local_tractography_toolbox>
{
public:
  explicit local_tractography_plugin(QWidget* parent = nullptr);

  void start() override;

private:
  void trace();

  std::array<std::size_t, 3> seed_offset() const;
  std::array<std::size_t, 3> seed_size  () const;
  std::array<std::size_t, 3> seed_stride() const;

  renderable*       streamline_renderer_;
  std::future<void> future_             ;
  bool              gpu_tracing_        = true ;
};
}

#endif
