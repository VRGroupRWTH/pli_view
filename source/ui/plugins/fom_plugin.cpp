#include /* implements */ <ui/plugins/fom_plugin.hpp>

#include <functional>
#include <future>
#include <limits>

#include <boost/optional.hpp>

#include <ui/plugins/data_plugin.hpp>
#include <ui/plugins/selector_plugin.hpp>
#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>
#include <visualization/vector_field.hpp>

namespace pli
{
fom_plugin::fom_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  line_edit_fiber_scale->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));

  connect(checkbox_enabled     , &QCheckBox::stateChanged    , [&] (int state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    vector_field_->set_active(state);
  });
  connect(slider_fiber_scale   , &QxtSpanSlider::valueChanged, [&](int value)
  {
    line_edit_fiber_scale->setText(QString::fromStdString(std::to_string(value)));
    update();
  });
  connect(line_edit_fiber_scale, &QLineEdit::editingFinished , [&]
  {
    logger_->info("Fiber scale is set to {}.", line_edit_fiber_scale->text().toStdString());
    update();
  });
}

void fom_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>    (), &data_plugin::on_change    , [&]
  {
    update();
  });
  connect(owner_->get_plugin<selector_plugin>(), &selector_plugin::on_change, [&]
  {
    update();
  });
  
  vector_field_ = owner_->viewer->add_renderable<vector_field>();

  logger_->info(std::string("Start successful."));
}
void fom_plugin::update() const
{
  return;

  logger_->info(std::string("Updating viewer..."));

  auto data_plugin     = owner_->get_plugin<pli::data_plugin>    ();
  auto selector_plugin = owner_->get_plugin<pli::selector_plugin>();
  auto io              = data_plugin    ->io    ();
  auto offset          = selector_plugin->offset();
  auto size            = selector_plugin->size  ();
  auto scale           = line_edit_utility::get_text<float>(line_edit_fiber_scale);

  if  (io == nullptr || size[0] == 0 || size[1] == 0 || size[2] == 0)
  {
    logger_->info(std::string("Update failed: No data."));
    return;
  }

  owner_->viewer->set_wait_spinner_enabled(true);

  std::array<float, 3>                          spacing    ;
  boost::optional<boost::multi_array<float, 3>> direction  ;
  boost::optional<boost::multi_array<float, 3>> inclination;

  std::future<void> result(std::async(std::launch::async, [&]()
  {
    try
    {
      if(checkbox_enabled)
      {
        spacing = io->load_vector_spacing();
        direction  .reset(io->load_fiber_direction_dataset  (offset, size));
        inclination.reset(io->load_fiber_inclination_dataset(offset, size));
      }
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  }));

  while(result.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();

  uint3  cuda_size    {unsigned(size[0]), unsigned(size[1]), unsigned(size[2])};
  float3 cuda_spacing {spacing[0], spacing[1], spacing[2]};
  if (direction.is_initialized() && direction.get().num_elements() > 0)
    vector_field_->set_data(cuda_size, direction.get().data(), inclination.get().data(), cuda_spacing, scale, [&](const std::string& message)
    {
      logger_->info(message);
    });

  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Update successful."));
}
}
