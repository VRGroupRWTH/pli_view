#include <pli_vis/ui/plugins/volume_rendering_plugin.hpp>

#include <boost/format.hpp>

#include <pli_vis/ui/application.hpp>
#include <pli_vis/utility/line_edit.hpp>
#include <pli_vis/utility/text_browser_sink.hpp>
#include <pli_vis/visualization/volume_renderer.hpp>

namespace pli
{
volume_rendering_plugin::volume_rendering_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);

  connect(checkbox_enabled   , &QCheckBox::stateChanged   , [&](bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    volume_renderer_->set_active(state);
  });
  connect(slider_step_size   , &QSlider::valueChanged     , [&]
  {
    auto step_size = float(slider_step_size->value()) / slider_step_size->maximum();
    line_edit_step_size->setText(QString::fromStdString((boost::format("%.4f") % step_size).str()));
  });
  connect(slider_step_size   , &QSlider::sliderReleased   , [&]
  {
    volume_renderer_->set_step_size(line_edit::get_text<double>(line_edit_step_size));
  });
  connect(line_edit_step_size, &QLineEdit::editingFinished, [&]
  {
    auto step_size = line_edit::get_text<double>(line_edit_step_size);
    slider_step_size->setValue(step_size * slider_step_size->maximum());
    volume_renderer_->set_step_size(step_size);
  });
}
void volume_rendering_plugin::start ()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>    (), &data_plugin::on_change             , [&]
  {
    upload();
  });
  connect(owner_->get_plugin<selector_plugin>(), &selector_plugin::on_change         , [&]
  {
    upload();
  });
  connect(transfer_function_editor             , &transfer_function_editor::on_change, [&]
  {
    volume_renderer_->set_transfer_function(transfer_function_editor->get_function());
  });
 
  auto step_size = double(slider_step_size->value()) / slider_step_size->maximum();

  volume_renderer_ = owner_->viewer->add_renderable<volume_renderer>();
  volume_renderer_->set_active           (checkbox_enabled->isChecked());
  volume_renderer_->set_step_size        (step_size);
  volume_renderer_->set_transfer_function(transfer_function_editor->get_function());

  line_edit_step_size->setText(QString::fromStdString((boost::format("%.4f") % step_size).str()));

  logger_->info(std::string("Start successful."));
}
void volume_rendering_plugin::upload()
{
  logger_->info(std::string("Updating viewer..."));

  auto io       = owner_->get_plugin<pli::data_plugin>    ()->io();
  auto selector = owner_->get_plugin<pli::selector_plugin>();
  auto offset   = selector->selection_offset();
  auto size     = selector->selection_size  ();
  auto stride   = selector->selection_stride();

  if  (io == nullptr || size[0] == 0 || size[1] == 0 || size[2] == 0)
  {
    logger_->info(std::string("Update failed: No data."));
    return;
  }

  size = {size[0] / stride[0], size[1] / stride[1], size[2] / stride[2]};

  owner_->viewer->set_wait_spinner_enabled(true);
  selector      ->setEnabled(false);

  // Load data from hard drive (on another thread).
  std::array<float, 3>                          spacing    ;
  boost::optional<boost::multi_array<float, 3>> retardation;
  std::vector<std::size_t>                      quantified_retardation(255);
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      spacing = io->load_vector_spacing();
      retardation.reset(io->load_retardation_dataset(offset, size, stride, false));
      std::for_each (retardation->origin(), retardation->origin() + retardation->num_elements(), 
      [&quantified_retardation] (const float& value)
      {
        quantified_retardation[value]++;
      });
      auto normalization_ratio = 255.0F / *std::max_element(quantified_retardation.begin(), quantified_retardation.end());
      std::transform(quantified_retardation.begin(), quantified_retardation.end(), quantified_retardation.begin(),
      [&normalization_ratio] (const std::size_t& value)
      {
        return float(value) * normalization_ratio;
      });
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  });
  while(future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();
    
  uint3  cuda_size   {unsigned(size[0]), unsigned(size[1]), unsigned(size[2])};
  float3 cuda_spacing{spacing[0], spacing[1], spacing[2]};
  volume_renderer_->set_data(cuda_size, cuda_spacing, retardation.get().data());

  transfer_function_editor->set_histogram_entries(quantified_retardation);

  selector      ->setEnabled(true);
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Update successful."));
}
}
