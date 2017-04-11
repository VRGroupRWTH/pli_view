#include /* implements */ <ui/plugins/scalar_plugin.hpp>

#include <functional>
#include <future>
#include <limits>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>
#include <visualization/scalar_field.hpp>

namespace pli
{
scalar_plugin::scalar_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  line_edit_offset_x->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_y->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_z->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_x  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_y  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_z  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  
  connect(checkbox_auto_update, &QCheckBox::stateChanged   , [&](int state)
  {
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
    logger_->info("Selection X offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_x));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_offset_y  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Y offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_y));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_offset_z  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Z offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_z));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_size_x    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection X size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_x));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_size_y    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Y size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_y));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_size_z    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Z size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_z));
    if (checkbox_auto_update->isChecked())
      update();
  });

  connect(checkbox_transmittance, &QCheckBox::stateChanged, [&] (int state)
  {
    logger_->info(std::string("Show transmittance set to ") + (state ? "true" : "false"));
    scalar_fields_["transmittance"]->set_active(state);
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(checkbox_retardation  , &QCheckBox::stateChanged, [&] (int state)
  {
    logger_->info(std::string("Show retardation set to ") + (state ? "true" : "false"));
    scalar_fields_["retardation"]->set_active(state);
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(checkbox_direction    , &QCheckBox::stateChanged, [&] (int state)
  {
    logger_->info(std::string("Show direction set to ") + (state ? "true" : "false"));
    scalar_fields_["direction"]->set_active(state);
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(checkbox_inclination  , &QCheckBox::stateChanged, [&] (int state)
  {
    logger_->info(std::string("Show inclination set to ") + (state ? "true" : "false"));
    scalar_fields_["inclination"]->set_active(state);
    if (checkbox_auto_update->isChecked())
      update();
  });
}

void scalar_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    logger_->info(std::string("Updating viewer."));
    update();
  });
  
  scalar_fields_["transmittance"] = owner_->viewer->add_renderable<scalar_field>();
  scalar_fields_["retardation"  ] = owner_->viewer->add_renderable<scalar_field>();
  scalar_fields_["direction"    ] = owner_->viewer->add_renderable<scalar_field>();
  scalar_fields_["inclination"  ] = owner_->viewer->add_renderable<scalar_field>();
}
void scalar_plugin::update() const
{
  auto data_plugin = owner_->get_plugin<pli::data_plugin>();
  auto io          = data_plugin->io();
  if (io == nullptr)
    return;

  auto load_transmittance = checkbox_transmittance->isChecked();
  auto load_retardation   = checkbox_retardation  ->isChecked();
  auto load_direction     = checkbox_direction    ->isChecked();
  auto load_inclination   = checkbox_inclination  ->isChecked();
  
  std::array<std::size_t, 3> offset = 
  { line_edit_utility::get_text<std::size_t>(line_edit_offset_x), 
    line_edit_utility::get_text<std::size_t>(line_edit_offset_y),
    line_edit_utility::get_text<std::size_t>(line_edit_offset_z)};     
  std::array<std::size_t, 3> size = 
  { line_edit_utility::get_text<std::size_t>(line_edit_size_x),
    line_edit_utility::get_text<std::size_t>(line_edit_size_y),
    line_edit_utility::get_text<std::size_t>(line_edit_size_z)};

  owner_->viewer->set_wait_spinner_enabled(true);

  std::array<float, 3>         spacing;
  boost::multi_array<float, 3> transmittance(boost::extents[size[0]][size[1]][size[2]]);
  boost::multi_array<float, 3> retardation  (boost::extents[size[0]][size[1]][size[2]]);
  boost::multi_array<float, 3> direction    (boost::extents[size[0]][size[1]][size[2]]);
  boost::multi_array<float, 3> inclination  (boost::extents[size[0]][size[1]][size[2]]);
  std::future<void> result(std::async(std::launch::async, [&]()
  {
    try
    {
      spacing = io->load_vector_spacing();
      if (load_transmittance)
        transmittance = io->load_transmittance_dataset    (offset, size, true);
      if (load_retardation)
        retardation   = io->load_retardation_dataset      (offset, size, true);
      if (load_direction)
        direction     = io->load_fiber_direction_dataset  (offset, size, true);
      if (load_inclination)
        inclination   = io->load_fiber_inclination_dataset(offset, size, true);
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
  scalar_fields_.at("transmittance")->set_data(cuda_size, transmittance.data(), cuda_spacing);
  scalar_fields_.at("retardation"  )->set_data(cuda_size, retardation  .data(), cuda_spacing);
  scalar_fields_.at("direction"    )->set_data(cuda_size, direction    .data(), cuda_spacing);
  scalar_fields_.at("inclination"  )->set_data(cuda_size, inclination  .data(), cuda_spacing);

  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();
}
}
