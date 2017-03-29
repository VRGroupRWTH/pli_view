#include /* implements */ <ui/plugins/retardation_plugin.hpp>

#include <limits>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>
#include <visualization/scalar_field.hpp>

namespace pli
{
retardation_plugin::retardation_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  line_edit_offset_x->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_y->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_z->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_x  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_y  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_z  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  
  connect(line_edit_offset_x, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection X offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_x));
    update();
  });
  connect(line_edit_offset_y, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Y offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_y));
    update();
  });
  connect(line_edit_offset_z, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Z offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_z));
    update();
  });
  connect(line_edit_size_x  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection X size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_x));
    update();
  });
  connect(line_edit_size_y  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Y size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_y));
    update();
  });
  connect(line_edit_size_z  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Z size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_z));
    update();
  });
  connect(checkbox_show     , &QCheckBox::stateChanged   , [&] (int state)
  {
    logger_->info(std::string("Show set to ") + (state ? "true" : "false"));
    update();
  });
}

void retardation_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    logger_->info(std::string("Updating viewer."));
    update();
  });
  
  scalar_field_ = owner_->viewer->add_renderable<scalar_field>();
}
void retardation_plugin::update() const
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

      auto retardation_map = io->load_retardation_map(offset, size);
      auto shape           = retardation_map.shape();
      auto spacing         = io->load_vector_spacing();
      
      scalar_field_->set_data(
        {unsigned(shape[0]), unsigned(shape[1]), unsigned(shape[2])},
        retardation_map.data(),
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
