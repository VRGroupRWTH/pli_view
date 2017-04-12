#include /* implements */ <ui/plugins/interactor_plugin.hpp>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>

namespace pli
{
interactor_plugin::interactor_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  line_edit_move_speed->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 8, this));
  line_edit_look_speed->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 8, this));

  connect(line_edit_move_speed, &QLineEdit::editingFinished, [&]
  {
    auto value = line_edit_utility::get_text<double>(line_edit_move_speed);
    logger_->info("Move speed is set to {}.", value);
    owner_->viewer->interactor()->set_move_speed(value);
  });
  connect(line_edit_look_speed, &QLineEdit::editingFinished, [&]
  {
    auto value = line_edit_utility::get_text<double>(line_edit_look_speed);
    logger_->info("Look speed is set to {}.", value);
    owner_->viewer->interactor()->set_look_speed(value);
  });

  connect(checkbox_ortho      , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Using {} projection.", state ? "orthographic" : "perspective");
    owner_->viewer->camera()->set_orthographic(state);
  });
  connect(button_reset        , &QPushButton::clicked      , [&]
  {
    logger_->info(std::string("Resetting camera transform."));
    owner_->viewer->camera()->set_translation({0, 0, 1});
    owner_->viewer->camera()->look_at        ({0, 0, 0});
  });
}

void interactor_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  auto interactor = owner_->viewer->interactor();
  interactor->set_move_speed(line_edit_utility::get_text<double>(line_edit_move_speed));
  interactor->set_look_speed(line_edit_utility::get_text<double>(line_edit_look_speed));

  logger_->info(std::string("Start successful."));
}
}
