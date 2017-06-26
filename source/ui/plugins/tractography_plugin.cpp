#include /* implements */ <ui/plugins/tractography_plugin.hpp>

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
  std::vector<float3> points;
  std::vector<float4> colors;
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      spacing = io->load_vector_spacing();
      unit_vectors.reset(io->load_fiber_unit_vectors_dataset(offset, size, stride, true));

      auto shape = unit_vectors.get().shape();
      tangent::CartesianGrid data(tangent::grid_dim_t{ { shape[0], shape[1], shape[2] } }, spacing);
      auto data_ptr = data.GetVectorPointer(0);
      for (auto x = 0; x < shape[0]; x++)
        for (auto y = 0; y < shape[1]; y++)
          for (auto z = 0; z < shape[2]; z++)
          {
            auto vector = unit_vectors.get()[x][y][z];
            data_ptr[z + shape[2] * (y + shape[1] * x)] = tangent::vector_t{ { vector[0], vector[1], vector[2] } };
          }

      // Seed with a 16x16x16 subgrid for now.
      std::vector<tangent::point_t> input;
      for (auto i = 0; i < 16; i++)
        for (auto j = 0; j < 16; j++)
          for (auto k = 0; k < 16; k++)
            input.push_back({{spacing[0] * i, spacing[1] * j, spacing[2] * k, 0.0F}});

      tangent::TraceRecorder recorder;
      tangent::SimpleCartGridStreamlineTracer tracer(&recorder);
      tracer.SetData(&data);
      tracer.SetIntegrationStep(0.001);
      tracer.SetNumberOfIterations(1000);
      auto output = tracer.TraceSeeds(input);

      auto& population = recorder.GetPopulation();
      for(auto i = 0; i < population.GetNumberOfTraces(); i++)
      {
        auto& path = population[i];
        for(auto j = 0; j < path.size() - 1; j++)
        {
          float3 start      = {path[j]  [0], path[j]    [1], path[j][2]};
          float3 end        = {path[j+1][0], path[j + 1][1], path[j][2]};
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
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Trace successful."));
}
}
