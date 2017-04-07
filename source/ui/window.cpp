#include /* implements */ <ui/window.hpp>

#include <ui/plugins/plugin.hpp>
#include <utility/qt_text_browser_sink.hpp>

namespace pli
{
window:: window()
{
  setupUi      (this);
  showMaximized();

  set_sink     (std::make_shared<qt_text_browser_sink>(console));
  bind_actions ();

  plugins_ = toolbox->findChildren<plugin*>(QRegExp("plugin")).toVector().toStdVector();
  for (auto plugin : plugins_)
    plugin->set_owner(this);
  for (auto plugin : plugins_)
    plugin->awake();
  for (auto plugin : plugins_)
    plugin->start();
}
window::~window()
{
  for (auto plugin : plugins_)
    plugin->destroy();
}

void window::bind_actions     ()
{  
  connect(action_fullscreen  , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Toggling fullscreen mode."));
    isFullScreen() ? showNormal() : showFullScreen();
  });
  connect(action_file_exit   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Closing window."));
    close();
  });
  connect(action_help_version, &QAction::triggered, [&] 
  {
    logger_->info(std::string("Version ") + __DATE__);
  });
}
}
