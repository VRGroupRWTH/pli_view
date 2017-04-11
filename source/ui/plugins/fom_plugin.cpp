#include /* implements */ <ui/plugins/fom_plugin.hpp>

#include <functional>
#include <future>
#include <limits>

#include <boost/optional.hpp>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>
#include <visualization/vector_field.hpp>

namespace pli
{
fom_plugin::fom_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  line_edit_offset_x->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_offset_y->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_offset_z->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_size_x  ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_size_y  ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_size_z  ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_scale   ->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));
  
  connect(checkbox_auto_update, &QCheckBox::stateChanged   , [&] (int state)
  {
    logger_->info("Auto update is {}.", state ? "enabled" : "disabled");
    button_update->setEnabled(!state);
    if (state)
      update();
  });
  connect(button_update       , &QPushButton::clicked      , [&]
  {
    update();
  });
  connect(line_edit_offset_x  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("X offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_x));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_offset_y  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Y offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_y));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_offset_z  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Z offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_z));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_size_x    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("X size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_x));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_size_y    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Y size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_y));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_size_z    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Z size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_z));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(checkbox_show       , &QCheckBox::stateChanged   , [&] (int state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    vector_field_->set_active(state);
  });
  connect(line_edit_scale     , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Scale is set to {}.", line_edit_scale->text().toStdString());
    if (checkbox_auto_update->isChecked())
      update();
  });
}

void fom_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    update();
  });
  
  vector_field_ = owner_->viewer->add_renderable<vector_field>();

  logger_->info(std::string("Start successful."));
}
void fom_plugin::update() const
{
  logger_->info(std::string("Updating viewer..."));

  auto data_plugin = owner_->get_plugin<pli::data_plugin>();
  auto io          = data_plugin->io();
  if  (io == nullptr)
  {
    logger_->info(std::string("Update failed: No data."));
    return;
  }

  owner_->viewer->set_wait_spinner_enabled(true);

  std::array<std::size_t, 3> offset =
  {line_edit_utility::get_text<std::size_t>(line_edit_offset_x),
   line_edit_utility::get_text<std::size_t>(line_edit_offset_y),
   line_edit_utility::get_text<std::size_t>(line_edit_offset_z)};
  std::array<std::size_t, 3> size =
  {line_edit_utility::get_text<std::size_t>(line_edit_size_x),
   line_edit_utility::get_text<std::size_t>(line_edit_size_y),
   line_edit_utility::get_text<std::size_t>(line_edit_size_z)};
  auto scale = line_edit_utility::get_text<float>(line_edit_scale);
   
  std::array<float, 3>                          spacing    ;
  boost::optional<boost::multi_array<float, 3>> direction  ;
  boost::optional<boost::multi_array<float, 3>> inclination;

  std::future<void> result(std::async(std::launch::async, [&]()
  {
    try
    {
      if(checkbox_show)
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
