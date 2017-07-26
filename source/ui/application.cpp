#include <pli_vis/ui/application.hpp>

#include <cuda_runtime_api.h>
#include <QShortcut>

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
  auto fullscreen_action = new QAction(this);
  fullscreen_action->setShortcut       (QKeySequence(Qt::Key_F11));
  fullscreen_action->setShortcutContext(Qt::ApplicationShortcut );
  addAction(fullscreen_action);
  connect(fullscreen_action, &QAction::triggered, this, [&]()
  {
    logger_->info(std::string("Toggling fullscreen mode."));
    isFullScreen() ? showNormal() : showFullScreen();
  });
  
  auto version_action = new QAction(this);
  version_action->setShortcut       (QKeySequence(Qt::Key_V));
  version_action->setShortcutContext(Qt::ApplicationShortcut );
  addAction(version_action);
  connect(version_action, &QAction::triggered, this, [&]()
  {
    logger_->info(std::string("Version ") + __DATE__);
  });

  auto gpu_info_action = new QAction(this);
  gpu_info_action->setShortcut       (QKeySequence(Qt::Key_G));
  gpu_info_action->setShortcutContext(Qt::ApplicationShortcut );
  addAction(gpu_info_action);
  connect(gpu_info_action, &QAction::triggered, this, [&]()
  {
    std::size_t free, total;
    cudaMemGetInfo(&free, &total);
    logger_->info("Available GPU memory: {} MB. Total GPU memory: {} MB.", free * 1E-6, total * 1E-6);
  });
}
}
