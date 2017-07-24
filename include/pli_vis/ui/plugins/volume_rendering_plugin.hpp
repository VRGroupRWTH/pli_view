#ifndef PLI_VIS_VOLUME_RENDERING_PLUGIN_HPP_
#define PLI_VIS_VOLUME_RENDERING_PLUGIN_HPP_

#include <future>

#include <pli_vis/ui/plugin.hpp>
#include <ui_volume_rendering_toolbox.h>

namespace pli
{
class volume_renderer;

class volume_rendering_plugin : public plugin<volume_rendering_plugin, Ui_volume_rendering_toolbox>
{
public:
  explicit volume_rendering_plugin(QWidget* parent = nullptr);

  void start () override;

private:
  void upload();
  
  volume_renderer*  volume_renderer_;
  std::future<void> future_         ;
};
}

#endif
