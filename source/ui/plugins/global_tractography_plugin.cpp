#include <pli_vis/ui/plugins/global_tractography_plugin.hpp>

#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
global_tractography_plugin::global_tractography_plugin(QWidget* parent) : plugin(parent)
{
  connect(checkbox_enabled      , &QCheckBox::stateChanged, [&] (bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    streamline_renderer_->set_active(state);
  });
  connect(button_trace_selection, &QPushButton::clicked   , [&]
  {
    trace();
  });
}

void global_tractography_plugin::start()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));
  streamline_renderer_ = owner_->viewer->add_renderable<streamline_renderer>();
}
void global_tractography_plugin::trace()
{
  owner_ ->set_is_loading(true);
  logger_->info          (std::string("Tracing..."));

  std::vector<float3> points    ;
  std::vector<float3> directions;
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      auto vectors = owner_->get_plugin<data_plugin>()->generate_vectors(true);
      auto shape   = vectors.shape();

      // TODO.
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  });
  while (future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();

  logger_->info          (std::string("Trace complete."));
  owner_ ->set_is_loading(false);
}
}
