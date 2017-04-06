#include /* implements */ <ui/plugins/scalar_plugin.hpp>

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
    logger_->info(std::string("Show set to ") + (state ? "true" : "false"));
    scalar_fields_["transmittance"]->set_active(state);
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(checkbox_retardation  , &QCheckBox::stateChanged, [&] (int state)
  {
    logger_->info(std::string("Show set to ") + (state ? "true" : "false"));
    scalar_fields_["retardation"]->set_active(state);
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(checkbox_direction    , &QCheckBox::stateChanged, [&] (int state)
  {
    logger_->info(std::string("Show set to ") + (state ? "true" : "false"));
    scalar_fields_["direction"]->set_active(state);
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(checkbox_inclination  , &QCheckBox::stateChanged, [&] (int state)
  {
    logger_->info(std::string("Show set to ") + (state ? "true" : "false"));
    scalar_fields_["inclination"]->set_active(state);
    if (checkbox_auto_update->isChecked())
      update();
  });
}

void scalar_plugin::start()
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
  try
  {
    auto data_plugin = owner_->get_plugin<pli::data_plugin>();
    auto io          = data_plugin->io();
    if (io == nullptr)
      return;

    auto spacing = io->load_vector_spacing();

    std::array<std::size_t, 3> offset = 
    { line_edit_utility::get_text<std::size_t>(line_edit_offset_x), 
      line_edit_utility::get_text<std::size_t>(line_edit_offset_y),
      line_edit_utility::get_text<std::size_t>(line_edit_offset_z)};
      
    std::array<std::size_t, 3> size = 
    { line_edit_utility::get_text<std::size_t>(line_edit_size_x),
      line_edit_utility::get_text<std::size_t>(line_edit_size_y),
      line_edit_utility::get_text<std::size_t>(line_edit_size_z)};

    if (checkbox_transmittance->isChecked())
    {
      auto scalar_map = io->load_transmittance_dataset(offset, size);
      auto shape      = scalar_map.shape();
      scalar_fields_.at("transmittance")->set_data(
        {unsigned(shape[0]), unsigned(shape[1]), unsigned(shape[2])},
        scalar_map.data(),
        {spacing[0], spacing[1], spacing[2]});
    }
    if (checkbox_retardation->isChecked())
    {
      auto scalar_map = io->load_retardation_dataset(offset, size);
      auto shape      = scalar_map.shape();
      scalar_fields_.at("retardation")->set_data(
        {unsigned(shape[0]), unsigned(shape[1]), unsigned(shape[2])},
        scalar_map.data(),
        {spacing[0], spacing[1], spacing[2]});
    }
    if (checkbox_direction->isChecked())
    {
      auto scalar_map = io->load_fiber_direction_dataset(offset, size);
      auto shape      = scalar_map.shape();
      scalar_fields_.at("direction")->set_data(
        {unsigned(shape[0]), unsigned(shape[1]), unsigned(shape[2])},
        scalar_map.data(),
        {spacing[0], spacing[1], spacing[2]});
    }
    if (checkbox_inclination->isChecked())
    {
      auto scalar_map = io->load_fiber_inclination_dataset(offset, size);
      auto shape      = scalar_map.shape();
      scalar_fields_.at("inclination")->set_data(
        {unsigned(shape[0]), unsigned(shape[1]), unsigned(shape[2])},
        scalar_map.data(),
        {spacing[0], spacing[1], spacing[2]});
    }
    
    owner_->viewer->update();
  }
  catch (std::exception& exception)
  {
    logger_->error(std::string(exception.what()));
  }
}
}
