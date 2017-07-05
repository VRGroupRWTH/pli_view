#ifndef PLI_VIS_VOLUME_RENDERING_PLUGIN_HPP_
#define PLI_VIS_VOLUME_RENDERING_PLUGIN_HPP_

#include <future>

#include <aspects/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_volume_rendering_toolbox.h>

namespace pli
{
class volume_renderer;

class volume_rendering_plugin : 
  public plugin, 
  public loggable<volume_rendering_plugin>,
  public Ui_volume_rendering_toolbox
{
public:
  volume_rendering_plugin(QWidget* parent = nullptr);
  void start () override;

private:
  void upload();
  
  volume_renderer*  volume_renderer_;
  std::future<void> future_         ;
};
}

#endif
