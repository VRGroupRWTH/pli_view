#include <pli_vis/ui/plugins/volume_rendering_plugin.hpp>

#include <boost/format.hpp>
#include <vector_functions.hpp>

#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>
#include <pli_vis/visualization/algorithms/volume_renderer.hpp>

namespace pli
{
volume_rendering_plugin::volume_rendering_plugin(QWidget* parent) : plugin(parent)
{
  line_edit_step_size->setText(QString::fromStdString((boost::format("%.4f") % (double(slider_step_size->value()) / slider_step_size->maximum())).str()));

  connect(checkbox_enabled        , &QCheckBox::stateChanged            , [&](bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    volume_renderer_->set_active(state);
  });
  connect(slider_step_size        , &QSlider::valueChanged              , [&]
  {
    line_edit_step_size->setText(QString::fromStdString((boost::format("%.4f") % (float(slider_step_size->value()) / slider_step_size->maximum())).str()));
  });
  connect(slider_step_size        , &QSlider::sliderReleased            , [&]
  {
    volume_renderer_->set_step_size(line_edit::get_text<double>(line_edit_step_size));
  });
  connect(line_edit_step_size     , &QLineEdit::editingFinished         , [&]
  {
    auto step_size = line_edit::get_text<double>(line_edit_step_size);
    slider_step_size->setValue(step_size * slider_step_size->maximum());
    volume_renderer_->set_step_size(step_size);
  });
  connect(transfer_function_editor, &transfer_function_editor::on_change, [&]
  {
    volume_renderer_->set_transfer_function(transfer_function_editor->get_function());
  });
}

void volume_rendering_plugin::start ()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_load, [&]
  {
    upload();
  });
  
  auto step_size = double(slider_step_size->value()) / slider_step_size->maximum();
  volume_renderer_ = owner_->viewer->add_renderable<volume_renderer>();
  volume_renderer_->set_active           (checkbox_enabled->isChecked());
  volume_renderer_->set_step_size        (step_size);
  volume_renderer_->set_transfer_function(transfer_function_editor->get_function());
}
void volume_rendering_plugin::upload()
{
  logger_->info(std::string("Updating viewer..."));

  auto& retardation = owner_->get_plugin<data_plugin>()->retardation();

  std::vector<std::size_t> histogram(255);
  std::for_each(retardation.origin(), retardation.origin() + retardation.num_elements(),
  [&histogram] (const float& value)
  {
    histogram[value]++;
  });

  auto ratio = 255.0F / *std::max_element(histogram.begin(), histogram.end());
  std::transform(histogram.begin(), histogram.end(), histogram.begin(),
  [&ratio](const std::size_t& value)
  {
    return float(value) * ratio;
  });

  volume_renderer_        ->set_data             (make_uint3(retardation.shape()[0], retardation.shape()[1], retardation.shape()[2]), retardation.data());
  transfer_function_editor->set_histogram_entries(histogram);

  logger_->info(std::string("Update successful."));
}
}
