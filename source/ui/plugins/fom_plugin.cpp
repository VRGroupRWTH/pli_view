#include <pli_vis/ui/plugins/fom_plugin.hpp>

#include <limits>

#include <boost/format.hpp>
#include <vector_functions.hpp>

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

  connect(checkbox_enabled                    , &QCheckBox::stateChanged      , [&] (bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    vector_field_->set_active(state);
  });
  connect(checkbox_view_dependent_transparency, &QCheckBox::stateChanged      , [&] (bool state)
  {
    logger_->info(std::string("View dependent transparency " + state ? "enabled." : "disabled."));
    vector_field_->set_view_dependent_transparency(state);
  });
  connect(slider_fiber_scale                  , &QxtSpanSlider::valueChanged  , [&]
  {
    line_edit_fiber_scale->setText(QString::fromStdString((boost::format("%.4f") % (float(slider_fiber_scale->value()) / slider_fiber_scale->maximum())).str()));
  });
  connect(slider_fiber_scale                  , &QxtSpanSlider::sliderReleased, [&]
  {
    upload();
  });
  connect(line_edit_fiber_scale               , &QLineEdit::editingFinished   , [&]
  {
    slider_fiber_scale->setValue(line_edit::get_text<double>(line_edit_fiber_scale) * slider_fiber_scale->maximum());
    upload();
  });
}

void fom_plugin::start ()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  vector_field_ = owner_->viewer->add_renderable<vector_field>();

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_load, [&]
  {
    upload();
  });
}
void fom_plugin::upload()
{
  logger_->info(std::string("Updating viewer..."));

  auto vectors = owner_->get_plugin<data_plugin>()->generate_vectors(true);
  vector_field_->set_data(
    make_uint3(vectors.shape()[0], vectors.shape()[1], vectors.shape()[2]),
    vectors.data(), 
    line_edit::get_text<float>(line_edit_fiber_scale), 
    [&] (const std::string& message) { logger_->info(message); });

  logger_->info(std::string("Update successful."));
}
}
