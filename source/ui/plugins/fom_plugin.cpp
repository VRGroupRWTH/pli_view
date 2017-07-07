#include <pli_vis/ui/plugins/fom_plugin.hpp>

#include <limits>

#include <boost/format.hpp>
#include <boost/optional.hpp>

#include <pli_vis/ui/plugins/data_plugin.hpp>
#include <pli_vis/ui/plugins/selector_plugin.hpp>
#include <pli_vis/ui/application.hpp>
#include <pli_vis/utility/line_edit_utility.hpp>
#include <pli_vis/utility/qt_text_browser_sink.hpp>
#include <pli_vis/visualization/vector_field.hpp>

namespace pli
{
fom_plugin::fom_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  line_edit_fiber_scale->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));

  connect(checkbox_enabled     , &QCheckBox::stateChanged      , [&] (bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    vector_field_->set_active(state);
  });
  connect(slider_fiber_scale   , &QxtSpanSlider::valueChanged  , [&]
  {
    auto scale = float(slider_fiber_scale->value()) / slider_fiber_scale->maximum();
    line_edit_fiber_scale->setText(QString::fromStdString((boost::format("%.4f") % scale).str()));
  });
  connect(slider_fiber_scale   , &QxtSpanSlider::sliderReleased, [&]
  {
    upload();
  });
  connect(line_edit_fiber_scale, &QLineEdit::editingFinished   , [&]
  {
    auto scale = line_edit_utility::get_text<double>(line_edit_fiber_scale);
    slider_fiber_scale->setValue(scale * slider_fiber_scale->maximum());
    upload();
  });
}

void fom_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>    (), &data_plugin::on_change    , [&]
  {
    upload();
  });
  connect(owner_->get_plugin<selector_plugin>(), &selector_plugin::on_change, [&]
  {
    upload();
  });
  
  vector_field_ = owner_->viewer->add_renderable<vector_field>();

  logger_->info(std::string("Start successful."));
}
void fom_plugin::upload()
{
  logger_->info(std::string("Updating viewer..."));

  auto io       = owner_->get_plugin<pli::data_plugin>    ()->io();
  auto selector = owner_->get_plugin<pli::selector_plugin>();
  auto offset   = selector->selection_offset();
  auto size     = selector->selection_size  ();
  auto stride   = selector->selection_stride();
  auto scale    = line_edit_utility::get_text<float>(line_edit_fiber_scale);

  if  (io == nullptr || size[0] == 0 || size[1] == 0 || size[2] == 0)
  {
    logger_->info(std::string("Update failed: No data."));
    return;
  }

  size = { size[0] / stride[0], size[1] / stride[1], size[2] / stride[2] };

  owner_->viewer->set_wait_spinner_enabled(true);
  selector->setEnabled(false);

  // Load data from hard drive (on another thread).
  std::array<float, 3>                          spacing    ;
  boost::optional<boost::multi_array<float, 3>> direction  ;
  boost::optional<boost::multi_array<float, 3>> inclination;
  boost::optional<boost::multi_array<float, 4>> unit_vector;
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      spacing = io->load_vector_spacing();
      direction  .reset(io->load_fiber_direction_dataset   (offset, size, stride, false));
      inclination.reset(io->load_fiber_inclination_dataset (offset, size, stride, false));
    }
    catch (std::exception& exception)
    {
      try
      {
        logger_->warn(std::string("Unable to load direction maps. Attempting to load unit vectors instead."));
        unit_vector.reset(io->load_fiber_unit_vectors_dataset(offset, size, stride, false));
      }
      catch (std::exception& exception2)
      {
        logger_->error(std::string(exception2.what()));
      }
    }
  });
  while(future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();

  // Upload data to GPU.
  uint3  cuda_size    {unsigned(size[0]), unsigned(size[1]), unsigned(size[2])};
  float3 cuda_spacing {spacing[0], spacing[1], spacing[2]};
  if (direction.is_initialized() && direction.get().num_elements() > 0)
    vector_field_->set_data(
      cuda_size, 
      direction  .get().data(), 
      inclination.get().data(), 
      cuda_spacing, 
      scale, 
      [&] (const std::string& message) { logger_->info(message); });
  if (unit_vector.is_initialized() && unit_vector.get().num_elements() > 0)
    vector_field_->set_data(
      cuda_size,
      reinterpret_cast<float3*>(unit_vector.get().data()),
      cuda_spacing,
      scale,
      [&] (const std::string& message) { logger_->info(message); });

  selector->setEnabled(true);
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Update successful."));
}
}
