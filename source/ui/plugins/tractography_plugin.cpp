#include /* implements */ <ui/plugins/tractography_plugin.hpp>

#include <boost/format.hpp>
#include <tangent-base/default_tracers.hpp>

#include <sh/vector_ops.h>
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

  line_edit_integration_step->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));
  line_edit_iterations      ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));

  connect(checkbox_enabled          , &QCheckBox::stateChanged      , [&] (bool state)
  {
    logger_      ->info(std::string(state ? "Enabled." : "Disabled."));
    basic_tracer_->set_active(state);
  });
  connect(slider_integration_step   , &QxtSpanSlider::valueChanged  , [&]
  {
    auto scale = float(slider_integration_step->value()) / slider_integration_step->maximum();
    line_edit_integration_step->setText(QString::fromStdString((boost::format("%.4f") % scale).str()));
  });
  connect(line_edit_integration_step, &QLineEdit::editingFinished   , [&]
  {
    auto scale = line_edit_utility::get_text<double>(line_edit_integration_step);
    slider_integration_step->setValue(scale * slider_integration_step->maximum());
  });
  connect(slider_iterations         , &QxtSpanSlider::valueChanged  , [&]
  {
    line_edit_iterations->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_iterations->value())));
  });
  connect(line_edit_iterations, &QLineEdit::editingFinished   , [&]
  {
    slider_iterations->setValue(line_edit_utility::get_text<int>(line_edit_iterations));
  });

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
  button_trace_selection->setEnabled(false);
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
  std::vector<float3> points;
  std::vector<float4> colors;
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      spacing = io->load_vector_spacing();
      unit_vectors.reset(io->load_fiber_unit_vectors_dataset(offset, size, stride, true));

      auto shape = unit_vectors.get().shape();
      tangent::CartesianGrid data(tangent::grid_dim_t{{shape[0], shape[1], shape[2]}}, spacing);
      std::vector<tangent::point_t> seeds;
      auto data_ptr = data.GetVectorPointer(0);
      for (auto x = 0; x < shape[0]; x++)
        for (auto y = 0; y < shape[1]; y++)
          for (auto z = 0; z < shape[2]; z++)
          {
            auto vector = unit_vectors.get()[x][y][z];
            data_ptr[x + shape[0] * (y + shape[1] * z)] = tangent::vector_t{{vector[0], vector[1], vector[2]}};
            seeds.push_back({{spacing[0] * x, spacing[1] * y, spacing[2] * z, 0.0F}});
          }

      tangent::TraceRecorder recorder;
      tangent::OmpCartGridStreamlineTracer tracer(&recorder);
      tracer.SetData(&data);
      tracer.SetIntegrationStep(0.001);
      tracer.SetNumberOfIterations(1000);
      auto output = tracer.TraceSeeds(seeds);

      auto& population = recorder.GetPopulation();
      for(auto i = 0; i < population.GetNumberOfTraces(); i++)
      {
        auto& path = population[i];
        for(auto j = 0; j < path.size() - 1; j++)
        {
          float3 start      = {path[j]    [0], path[j]    [1], path[j]    [2]};
          float3 end        = {path[j + 1][0], path[j + 1][1], path[j + 1][2]};
          auto   difference = fabs(end - start);
          auto   normalized = normalize(difference);
          points.push_back(start);
          points.push_back(end  );
          colors.push_back(float4{normalized.x, normalized.y, normalized.z, 1.0});
          colors.push_back(float4{normalized.x, normalized.y, normalized.z, 1.0});
        }
      }
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  });
  while (future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();

  if (unit_vectors.is_initialized() && unit_vectors.get().num_elements() > 0)
    basic_tracer_->set_data(points, colors);
  else
  {
    logger_->info(std::string("Trace failed: Tractography only supports unit vectors (MSA-0309 style) at the moment."));
    return;
  }

  selector->setEnabled(true);
  button_trace_selection->setEnabled(true);
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Trace successful."));
}
}
