#include <pli_vis/ui/plugins/interactor_plugin.hpp>

#include <boost/format.hpp>

#include <pli_vis/ui/window.hpp>
#include <pli_vis/utility/line_edit_utility.hpp>
#include <pli_vis/utility/qt_text_browser_sink.hpp>

namespace pli
{
interactor_plugin::interactor_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  line_edit_move_speed->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 4, this));
  line_edit_look_speed->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 4, this));

  connect(slider_move_speed   , &QSlider::valueChanged     , [&]
  {
    auto speed = float(slider_move_speed->value()) / slider_move_speed->maximum();
    line_edit_move_speed->setText(QString::fromStdString((boost::format("%.4f") % speed).str()));
    owner_->viewer->interactor()->set_move_speed(speed);
  });
  connect(slider_look_speed   , &QSlider::valueChanged     , [&]
  {
    auto speed = float(slider_look_speed->value()) / slider_look_speed->maximum();
    line_edit_look_speed->setText(QString::fromStdString((boost::format("%.4f") % speed).str()));
    owner_->viewer->interactor()->set_look_speed(speed);
  });
  connect(line_edit_move_speed, &QLineEdit::editingFinished, [&]
  {
    auto speed = line_edit_utility::get_text<double>(line_edit_move_speed);
    slider_move_speed->setValue(speed * slider_move_speed->maximum());
    owner_->viewer->interactor()->set_move_speed(speed);
  });
  connect(line_edit_look_speed, &QLineEdit::editingFinished, [&]
  {
    auto speed = line_edit_utility::get_text<double>(line_edit_look_speed);
    slider_look_speed->setValue(speed * slider_look_speed->maximum());
    owner_->viewer->interactor()->set_look_speed(speed);
  });
  connect(button_reset_camera , &QPushButton::clicked      , [&]
  {
    logger_->info(std::string("Resetting camera transform."));
    owner_->viewer->camera()->set_translation({0, 0, 1});
    owner_->viewer->camera()->look_at        ({0, 0, 0});
  });
}

void interactor_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  auto move_value = double(slider_move_speed->value()) / slider_move_speed->maximum();
  auto look_value = double(slider_look_speed->value()) / slider_look_speed->maximum();

  line_edit_move_speed->setText(QString::fromStdString((boost::format("%.4f") % move_value).str()));
  line_edit_look_speed->setText(QString::fromStdString((boost::format("%.4f") % look_value).str()));

  auto interactor = owner_->viewer->interactor();
  interactor->set_move_speed(move_value);
  interactor->set_look_speed(look_value);

  logger_->info(std::string("Start successful."));
}
}
