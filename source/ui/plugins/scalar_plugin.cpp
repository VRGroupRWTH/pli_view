#include <pli_vis/ui/plugins/scalar_plugin.hpp>

#include <boost/optional.hpp>

#include <pli_vis/ui/plugins/data_plugin.hpp>
#include <pli_vis/ui/plugins/selector_plugin.hpp>
#include <pli_vis/ui/window.hpp>
#include <pli_vis/utility/qt_text_browser_sink.hpp>
#include <pli_vis/visualization/scalar_field.hpp>

namespace pli
{
scalar_plugin::scalar_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  connect(checkbox_enabled      , &QCheckBox::stateChanged, [&](bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    scalar_field_->set_active(state);
  });
  connect(checkbox_transmittance, &QRadioButton::clicked  , [&]()
  {
    logger_->info(std::string("Transmittance maps are selected."));
    upload();
  });
  connect(checkbox_retardation  , &QRadioButton::clicked  , [&]()
  {
    logger_->info(std::string("Retardation maps are selected."));
    upload();
  });
  connect(checkbox_direction    , &QRadioButton::clicked  , [&]()
  {
    logger_->info(std::string("Direction maps are selected."));
    upload();
  });
  connect(checkbox_inclination  , &QRadioButton::clicked  , [&]()
  {
    logger_->info(std::string("Inclination maps are selected."));
    upload();
  });
}

void scalar_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>    (), &data_plugin    ::on_change, [&]
  {
    upload();
  });
  connect(owner_->get_plugin<selector_plugin>(), &selector_plugin::on_change, [&]
  {
    upload();
  });
  
  scalar_field_ = owner_->viewer->add_renderable<scalar_field>();

  logger_->info(std::string("Start successful."));
}
void scalar_plugin::upload()
{
  logger_->info(std::string("Updating viewer..."));

  auto io       = owner_->get_plugin<pli::data_plugin>    ()->io();
  auto selector = owner_->get_plugin<pli::selector_plugin>();
  auto offset   = selector->selection_offset();
  auto size     = selector->selection_size  ();
  auto stride   = selector->selection_stride();

  if (io == nullptr || size[0] == 0 || size[1] == 0 || size[2] == 0)
  {
    logger_->info(std::string("Update failed: No data."));
    return;
  }

  size = {size[0] / stride[0], size[1] / stride[1], 1};

  owner_->viewer->set_wait_spinner_enabled(true);
  selector->setEnabled(false);

  // Load data from hard drive (on another thread).
  std::array<float, 3>                          spacing;
  boost::optional<boost::multi_array<float, 3>> data   ;
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      spacing = io->load_vector_spacing();
      if (checkbox_transmittance->isChecked()) data.reset(io->load_transmittance_dataset    (offset, size, stride));
      if (checkbox_retardation  ->isChecked()) data.reset(io->load_retardation_dataset      (offset, size, stride));
      if (checkbox_direction    ->isChecked()) data.reset(io->load_fiber_direction_dataset  (offset, size, stride));
      if (checkbox_inclination  ->isChecked()) data.reset(io->load_fiber_inclination_dataset(offset, size, stride));
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  });
  while(future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();

  // Upload data to GPU.
  uint3  cuda_size    {unsigned(size[0]), unsigned(size[1]), unsigned(size[2])};
  float3 cuda_spacing {spacing[0], spacing[1], spacing[2]};
  if (data.is_initialized() && data.get().num_elements() > 0)
    scalar_field_->set_data(cuda_size, data.get().data(), cuda_spacing);

  selector->setEnabled(true);
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Update successful."));
}
}
