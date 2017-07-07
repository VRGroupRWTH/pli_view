#include /* implements */ <ui/plugins/tractography_plugin.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

#include <cuda/sh/convert.h>
#include <ui/plugins/data_plugin.hpp>
#include <ui/plugins/selector_plugin.hpp>
#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>

namespace pli
{
tractography_plugin::tractography_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  connect(button_trace_selection, &QPushButton::clicked, [&]
  {
    trace();
  });
}

void tractography_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  logger_->info(std::string("Start successful."));
}
void tractography_plugin::trace()
{
  logger_->info(std::string("Tracing..."));

  try
  {
    auto data_plugin     = owner_->get_plugin<pli::data_plugin>();
    auto io              = data_plugin->io();
    if(io == nullptr)
    {
      logger_->info(std::string("Trace failed: No data."));
      return;
    }

    auto selector_plugin = owner_->get_plugin<pli::selector_plugin>();
    auto offset          = selector_plugin->selection_offset();
    auto size            = selector_plugin->selection_size  ();
      
    auto fiber_direction_map   = io->load_fiber_direction_dataset  (offset, size);
    auto fiber_inclination_map = io->load_fiber_inclination_dataset(offset, size);
    auto spacing               = io->load_vector_spacing();
    auto shape                 = fiber_direction_map.shape();
    
    // TODO!

    owner_->viewer->update();

    logger_->info(std::string("Trace successful."));
  }
  catch (std::exception& exception)
  {
    logger_->error(std::string(exception.what()));
  }
}
}
