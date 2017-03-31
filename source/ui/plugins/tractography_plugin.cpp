#include /* implements */ <ui/plugins/tractography_plugin.hpp>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>

namespace pli
{
tractography_plugin::tractography_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  line_edit_offset_x->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_offset_y->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_offset_z->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_size_x  ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_size_y  ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_size_z  ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  
  connect(line_edit_offset_x  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents X offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_x));
  });
  connect(line_edit_offset_y  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents Y offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_y));
  });
  connect(line_edit_offset_z  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents Z offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_z));
  });
  connect(line_edit_size_x    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents X size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_x));
  });
  connect(line_edit_size_y    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents Y size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_y));
  });
  connect(line_edit_size_z    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents Z size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_z));
  });
  connect(button_trace        , &QPushButton::clicked      , [&]
  {
    update();
  });
}

void tractography_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));
}
void tractography_plugin::update() const
{
  try
  {
    auto data_plugin = owner_->get_plugin<pli::data_plugin>();
    auto io          = data_plugin->io();

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
    
    // TODO: Trace the directions and inclinations.

    owner_->viewer->update();
  }
  catch (std::exception& exception)
  {
    logger_->error(std::string(exception.what()));
  }
}
}
