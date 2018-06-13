#include <pli_vis/ui/plugins/fom_plugin.hpp>

#include <limits>

#include <boost/format.hpp>
#include <QDoubleValidator>
#include <vector_functions.hpp>

#include <pli_vis/ui/plugins/color_plugin.hpp>
#include <pli_vis/ui/plugins/data_plugin.hpp>
#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>
#include <pli_vis/visualization/algorithms/vector_field.hpp>

namespace pli
{
fom_plugin::fom_plugin(QWidget* parent) : plugin(parent)
{
  line_edit_fiber_scale->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));

  line_edit_rate_of_decay->setText(QString::fromStdString(std::to_string(slider_rate_of_decay->value())));

  connect(checkbox_enabled                    , &QCheckBox::stateChanged      , [&] (bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    vector_field_->set_active(state);
  });
  connect(slider_fiber_scale                  , &QxtSpanSlider::valueChanged  , [&]
  {
    line_edit_fiber_scale->setText(QString::fromStdString((boost::format("%.4f") % (float(slider_fiber_scale->value()) / slider_fiber_scale->maximum())).str()));
  });
  connect(slider_fiber_scale                  , &QxtSpanSlider::sliderReleased, [&]
  {
    vector_field_->set_scale(line_edit::get_text<float>(line_edit_fiber_scale));
  });
  connect(line_edit_fiber_scale               , &QLineEdit::editingFinished   , [&]
  {
    auto value = line_edit::get_text<float>(line_edit_fiber_scale);
    slider_fiber_scale->setValue (value * slider_fiber_scale->maximum());
    vector_field_     ->set_scale(value);
  });
  connect(checkbox_view_dependent             , &QCheckBox::stateChanged      , [&] (bool state)
  {
    logger_->info(std::string("View dependent transparency is " + state ? "enabled." : "disabled."));
    vector_field_->set_view_dependent_transparency(state);
    label_rate_of_decay    ->setEnabled(state);
    slider_rate_of_decay   ->setEnabled(state);
    line_edit_rate_of_decay->setEnabled(state);
  });
  connect(slider_rate_of_decay                , &QxtSpanSlider::valueChanged  , [&]
  {
    line_edit_rate_of_decay->setText(QString::fromStdString(std::to_string(slider_rate_of_decay->value())));
    vector_field_->set_view_dependent_rate_of_decay(slider_rate_of_decay->value());
  });
  connect(line_edit_rate_of_decay             , &QLineEdit::editingFinished   , [&]
  {
    auto value = line_edit::get_text<int>(line_edit_rate_of_decay);
    slider_rate_of_decay->setValue(value);
    vector_field_->set_view_dependent_rate_of_decay(value);
  });
}

void fom_plugin::start ()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  vector_field_ = owner_->viewer->add_renderable<vector_field>();

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_load    , [&]
  {
    upload();
  });
  connect(owner_->get_plugin<color_plugin>(), &color_plugin::on_change, [&] (int mode, float k, bool inverted)
  {
    vector_field_->set_color_mapping(mode, k, inverted);
  });

}
void fom_plugin::upload()
{
  logger_->info(std::string("Updating viewer..."));

  auto vectors = owner_->get_plugin<data_plugin>()->generate_vectors(true);
  vector_field_->set_data(
    make_uint3(vectors.shape()[0], vectors.shape()[1], vectors.shape()[2]),
    1,
    vectors.data(),
    [&] (const std::string& message) { logger_->info(message); });

  logger_->info(std::string("Update successful."));
}
}
