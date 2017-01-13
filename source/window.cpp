#include /* implements */ <window.hpp>

#include <QFileDialog>

#include <iostream>

namespace pli
{
window:: window()
{
  ui_.setupUi(this);
  bind_actions();
  showMaximized();
}
window::~window()
{
  
}

void window::bind_actions()
{
  connect(ui_.action_file_open, &QAction::triggered, [&] {
    logger_->info(std::string("Opening file dialog."));
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("PLI Files (*.h5)"));
    logger_->info("Closing file dialog. Selection: {}.", filename.toStdString());
  });
  connect(ui_.action_file_exit, &QAction::triggered, [&] {
    logger_->info(std::string("Closing window."));
    close();
  });
  connect(ui_.action_edit_undo, &QAction::triggered, [&] {
    logger_->info(std::string("Undoing last action."));
  });
  connect(ui_.action_edit_redo, &QAction::triggered, [&] {
    logger_->info(std::string("Redoing last action."));
  });
  connect(ui_.action_help_version, &QAction::triggered, [&] {
    logger_->info(std::string("Displaying version information."));
  });
}
}