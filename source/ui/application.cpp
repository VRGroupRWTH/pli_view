#include <pli_vis/ui/application.hpp>

#include <cuda_runtime_api.h>

#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/plugin_base.hpp>

namespace pli
{
application:: application()
{
  setupUi      (this);
  showMaximized();

  set_sink    (std::make_shared<text_browser_sink>(console));
  bind_actions();

  plugins_ = findChildren<plugin_base*>(QRegExp("plugin")).toVector().toStdVector();
  for (auto plugin : plugins_)
    plugin->set_owner(this);
  for (auto plugin : plugins_)
    plugin->awake();
  for (auto plugin : plugins_)
    plugin->start();

  splitter_vertical_left->setSizes(QList<int>{height(), 0});

  action_help_version ->trigger();
  action_help_gpu_info->trigger();
}
application::~application()
{
  for (auto plugin : plugins_)
    plugin->destroy();
}

void application::bind_actions()
{  
  connect(action_fullscreen   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Toggling fullscreen mode."));
    isFullScreen() ? showNormal() : showFullScreen();
  });
  connect(action_file_exit    , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Closing application."));
    close();
  });
  connect(action_help_version , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Version ") + __DATE__);
  });
  connect(action_help_gpu_info, &QAction::triggered, [&]
  {
    std::size_t free, total;
    cudaMemGetInfo(&free, &total);
    logger_->info("Available GPU memory: {} MB. Total GPU memory: {} MB.", free * 1E-6, total * 1E-6);
  });
}
}
