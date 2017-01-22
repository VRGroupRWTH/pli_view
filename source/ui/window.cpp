#include /* implements */ <ui/window.hpp>

#include <ui/plugins/plugin.hpp>
#include <utility/spdlog/qt_text_browser_sink.hpp>

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

  vtkObject::GlobalWarningDisplayOff();
}
window::~window()
{
  for (auto plugin : plugins_)
    plugin->destroy();
}

void window::bind_actions()
{  
  connect(action_file_exit   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Closing window."));
    close();
  });
  connect(action_edit_undo   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Undoing last action."));
    // TODO.
  });
  connect(action_edit_redo   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Redoing last action."));
    // TODO.
  });
  connect(action_help_version, &QAction::triggered, [&] 
  {
    logger_->info(std::string("Version 1.0."));
  });
}
}
