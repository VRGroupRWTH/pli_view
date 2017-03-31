#include /* implements */ <ui/plugins/fom_plugin.hpp>

#include <limits>

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
  connect(checkbox_show       , &QCheckBox::stateChanged   , [&] (int state)
  {
    logger_->info(std::string("Show set to ") + (state ? "true" : "false"));
    vector_field_->set_active(state);
  });
  connect(line_edit_scale     , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Vector scale set to " + line_edit_scale->text().toStdString());
    if (checkbox_auto_update->isChecked())
      update();
  });
}

void fom_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    logger_->info(std::string("Updating viewer."));
    update();
  });
  
  vector_field_ = owner_->viewer->add_renderable<vector_field>();
}
void fom_plugin::update() const
{
  try
  {
    auto data_plugin = owner_->get_plugin<pli::data_plugin>();
    auto io          = data_plugin->io();

    if (io && checkbox_show->isChecked())
    {
      std::array<std::size_t, 3> offset = 
      { line_edit_utility::get_text<std::size_t>(line_edit_offset_x), 
        line_edit_utility::get_text<std::size_t>(line_edit_offset_y),
        line_edit_utility::get_text<std::size_t>(line_edit_offset_z)};
      
      std::array<std::size_t, 3> size = 
      { line_edit_utility::get_text<std::size_t>(line_edit_size_x),
        line_edit_utility::get_text<std::size_t>(line_edit_size_y),
        line_edit_utility::get_text<std::size_t>(line_edit_size_z)};

      auto fiber_direction_map   = io->load_fiber_direction_map  (offset, size);
      auto fiber_inclination_map = io->load_fiber_inclination_map(offset, size);
      auto shape                 = fiber_direction_map.shape();
      auto spacing               = io->load_vector_spacing  ();
      auto scale                 = line_edit_utility::get_text<float>(line_edit_scale);
      
      vector_field_->set_data(
        {unsigned(shape[0]), unsigned(shape[1]), unsigned(shape[2])},
        fiber_direction_map  .data(), 
        fiber_inclination_map.data(),
        {spacing[0], spacing[1], spacing[2]},
        scale);
    }
    
    owner_->viewer->update();
  }
  catch (std::exception& exception)
  {
    logger_->error(std::string(exception.what()));
  }
}
}
