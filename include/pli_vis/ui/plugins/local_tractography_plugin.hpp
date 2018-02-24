#ifndef PLI_VIS_LOCAL_TRACTOGRAPHY_PLUGIN_HPP_
#define PLI_VIS_LOCAL_TRACTOGRAPHY_PLUGIN_HPP_

#include <cstddef>
#include <future>
#include <memory>
#include <vector>

#include <vector_types.h>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/ui/plugin.hpp>
#include <ui_local_tractography_toolbox.h>

namespace pli
{
class remote_viewer;
class local_tractography_plugin : public plugin<local_tractography_plugin, Ui_local_tractography_toolbox>
{
public:
  explicit local_tractography_plugin(QWidget* parent = nullptr);

  void start() override;
  
  std::array<std::size_t, 3> seed_offset() const;
  std::array<std::size_t, 3> seed_size  () const;
  std::array<std::size_t, 3> seed_stride() const;
  
  float       step      () const
  {
    return float(slider_integration_step->value()) / slider_integration_step->maximum();
  }
  std::size_t iterations() const
  {
    return slider_iterations->value();
  }

private:
  void trace();
  void remote_trace();

  std::vector<float4>            vertices_           ;
  std::vector<float4>            tangents_           ;
  std::vector<unsigned>          indices_            ;
  renderable*                    streamline_renderer_;
  std::future<void>              future_             ;
  bool                           gpu_tracing_        = false  ;
  std::unique_ptr<remote_viewer> remote_viewer_      = nullptr;
};
}

#endif
