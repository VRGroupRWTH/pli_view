#include <pli_vis/ui/plugins/interactor_plugin.hpp>

#include <boost/format.hpp>

#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>
#include <pli_vis/visualization/interactors/first_person_interactor.hpp>
#include <pli_vis/visualization/interactors/orbit_interactor.hpp>
#include <pli_vis/visualization/interactors/simple_interactor.hpp>

namespace pli
{
interactor_plugin::interactor_plugin(QWidget* parent) : plugin(parent)
{
  line_edit_move_speed->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 4, this));
  line_edit_look_speed->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 4, this));
  
  connect(radio_button_vtklike     , &QRadioButton::clicked     , [&]
  {
    logger_->info(std::string("VTK-like controls selected."));
    owner_->viewer->set_interactor<simple_interactor>();
    owner_->viewer->interactor()->set_move_speed(float(slider_move_speed->value()) / slider_move_speed->maximum());
    owner_->viewer->interactor()->set_look_speed(float(slider_look_speed->value()) / slider_look_speed->maximum());
  });
  connect(radio_button_orbit       , &QRadioButton::clicked     , [&]
  {
    logger_->info(std::string("Orbit controls selected."));
    owner_->viewer->set_interactor<orbit_interactor>();
    owner_->viewer->interactor()->set_move_speed(float(slider_move_speed->value()) / slider_move_speed->maximum());
    owner_->viewer->interactor()->set_look_speed(float(slider_look_speed->value()) / slider_look_speed->maximum());
  });
  connect(radio_button_wasd        , &QRadioButton::clicked     , [&]
  {
    logger_->info(std::string("WASD controls selected."));
    owner_->viewer->set_interactor<first_person_interactor>();
    owner_->viewer->interactor()->set_move_speed(float(slider_move_speed->value()) / slider_move_speed->maximum());
    owner_->viewer->interactor()->set_look_speed(float(slider_look_speed->value()) / slider_look_speed->maximum());
  });
  connect(radio_button_orthographic, &QRadioButton::clicked     , [&]
  {
    logger_->info(std::string("Orthographic projection selected."));
    owner_->viewer->camera()->set_orthographic(true);
    owner_->viewer->reset_camera_transform();
  });
  connect(radio_button_perspective , &QRadioButton::clicked     , [&]
  {
    logger_->info(std::string("Perspective projection selected."));
    owner_->viewer->camera()->set_orthographic(false);
    owner_->viewer->reset_camera_transform();
  });
  connect(slider_move_speed        , &QSlider::valueChanged     , [&]
  {
    auto speed = float(slider_move_speed->value()) / slider_move_speed->maximum();
    line_edit_move_speed->setText(QString::fromStdString((boost::format("%.4f") % speed).str()));
    owner_->viewer->interactor()->set_move_speed(speed);
  });
  connect(slider_look_speed        , &QSlider::valueChanged     , [&]
  {
    auto speed = float(slider_look_speed->value()) / slider_look_speed->maximum();
    line_edit_look_speed->setText(QString::fromStdString((boost::format("%.4f") % speed).str()));
    owner_->viewer->interactor()->set_look_speed(speed);
  });
  connect(line_edit_move_speed     , &QLineEdit::editingFinished, [&]
  {
    auto speed = line_edit::get_text<double>(line_edit_move_speed);
    slider_move_speed->setValue(speed * slider_move_speed->maximum());
    owner_->viewer->interactor()->set_move_speed(speed);
  });
  connect(line_edit_look_speed     , &QLineEdit::editingFinished, [&]
  {
    auto speed = line_edit::get_text<double>(line_edit_look_speed);
    slider_look_speed->setValue(speed * slider_look_speed->maximum());
    owner_->viewer->interactor()->set_look_speed(speed);
  });
  connect(button_reset_camera      , &QPushButton::clicked      , [&]
  {
    logger_->info(std::string("Resetting camera transform."));
    owner_->viewer->reset_camera_transform();
  });
}

void interactor_plugin::start()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  auto move_value = double(slider_move_speed->value()) / slider_move_speed->maximum();
  auto look_value = double(slider_look_speed->value()) / slider_look_speed->maximum();
  line_edit_move_speed->setText(QString::fromStdString((boost::format("%.4f") % move_value).str()));
  line_edit_look_speed->setText(QString::fromStdString((boost::format("%.4f") % look_value).str()));

  auto interactor = owner_->viewer->interactor();
  interactor->set_move_speed(move_value);
  interactor->set_look_speed(look_value);
}
}
