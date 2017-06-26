#include /* implements */ <ui/plugins/tractography_plugin.hpp>

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

  basic_tracer_ = owner_->viewer->add_renderable<basic_tracer>();

  logger_->info(std::string("Start successful."));
}
void tractography_plugin::trace()
{
  logger_->info(std::string("Tracing..."));

  auto io       = owner_->get_plugin<pli::data_plugin>()->io();
  auto selector = owner_->get_plugin<pli::selector_plugin>();
  auto offset   = selector->selection_offset();
  auto size     = selector->selection_size  ();
  auto stride   = selector->selection_stride();
      
  selector->setEnabled(false);
  owner_->viewer->set_wait_spinner_enabled(true);
  owner_->viewer->update();

  if(io == nullptr)
  {
    logger_->info(std::string("Trace failed: No data."));
    return;
  }

  // Load data from hard drive (on another thread).
  std::array<float, 3>                          spacing;
  boost::optional<boost::multi_array<float, 4>> unit_vectors;
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      spacing = io->load_vector_spacing();
      unit_vectors.reset(io->load_fiber_unit_vectors_dataset(offset, size, stride, false));
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  });
  while (future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();

  if (unit_vectors.is_initialized() && unit_vectors.get().num_elements() > 0)
    basic_tracer_->trace(unit_vectors.get());
  else
  {
    logger_->info(std::string("Trace failed: Tractography only supports unit vectors (MSA-0309 style) at the moment."));
    return;
  }

  selector->setEnabled(true);
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Trace successful."));
}
}
