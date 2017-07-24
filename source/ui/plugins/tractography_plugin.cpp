#include <pli_vis/ui/plugins/tractography_plugin.hpp>

#include <boost/format.hpp>
#include <tangent-base/default_tracers.hpp>

#include <pli_vis/cuda/sh/convert.h>
#include <pli_vis/cuda/sh/vector_ops.h>
#include <pli_vis/ui/plugins/data_plugin.hpp>
#include <pli_vis/ui/plugins/selector_plugin.hpp>
#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
tractography_plugin::tractography_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);

  line_edit_x               ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_y               ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_z               ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_integration_step->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));
  line_edit_iterations      ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
 
  line_edit_integration_step->setText(QString::fromStdString((boost::format("%.4f") % (float(slider_integration_step->value()) / slider_integration_step->maximum())).str()));
  line_edit_iterations      ->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_iterations->value())));
  line_edit_x               ->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_x         ->value())));
  line_edit_y               ->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_y         ->value())));
  line_edit_z               ->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_z         ->value())));

  connect(checkbox_enabled          , &QCheckBox::stateChanged    , [&] (bool state)
  {
    logger_      ->info(std::string(state ? "Enabled." : "Disabled."));
    streamline_renderer_->set_active(state);
  });
  connect(slider_x                  , &QSlider::valueChanged      , [&]
  {
    line_edit_x->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_x->value())));
  });
  connect(line_edit_x               , &QLineEdit::editingFinished , [&]
  {
    slider_x->setValue(line_edit::get_text<int>(line_edit_x));
  });
  connect(slider_y                  , &QSlider::valueChanged      , [&]
  {
    line_edit_y->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_y->value())));
  });
  connect(line_edit_y               , &QLineEdit::editingFinished , [&]
  {
    slider_y->setValue(line_edit::get_text<int>(line_edit_y));
  });
  connect(slider_z                  , &QSlider::valueChanged      , [&]
  {
    line_edit_z->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_z->value())));
  });
  connect(line_edit_z               , &QLineEdit::editingFinished , [&]
  {
    slider_z->setValue(line_edit::get_text<int>(line_edit_z));
  });
  connect(slider_integration_step   , &QSlider::valueChanged      , [&]
  {
    auto scale = float(slider_integration_step->value()) / slider_integration_step->maximum();
    line_edit_integration_step->setText(QString::fromStdString((boost::format("%.4f") % scale).str()));
  });
  connect(line_edit_integration_step, &QLineEdit::editingFinished , [&]
  {
    auto scale = line_edit::get_text<double>(line_edit_integration_step);
    slider_integration_step->setValue(scale * slider_integration_step->maximum());
  });
  connect(slider_iterations         , &QSlider::valueChanged      , [&]
  {
    line_edit_iterations->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_iterations->value())));
  });
  connect(line_edit_iterations      , &QLineEdit::editingFinished , [&]
  {
    slider_iterations->setValue(line_edit::get_text<int>(line_edit_iterations));
  });
  connect(button_trace_selection    , &QPushButton::clicked       , [&]
  {
    trace();
  });
}

void tractography_plugin::start()
{
  streamline_renderer_ = owner_->viewer->add_renderable<streamline_renderer>();

  set_sink(std::make_shared<text_browser_sink>(owner_->console));
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
      
  if(io == nullptr)
  {
    logger_->info(std::string("Trace failed: No data."));
    return;
  }

  size = { size[0] / stride[0], size[1] / stride[1], size[2] / stride[2] };

  selector->setEnabled(false);
  button_trace_selection->setEnabled(false);
  owner_->viewer->set_wait_spinner_enabled(true);
  owner_->viewer->update();

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
      auto data_ptr = data.GetVectorPointer(0);
      for (auto x = 0; x < shape[0]; x++)
        for (auto y = 0; y < shape[1]; y++)
          for (auto z = 0; z < shape[2]; z++)
          {
            auto vector = unit_vectors.get()[x][y][z];
            data_ptr[x + shape[0] * (y + shape[1] * z)] = tangent::vector_t{{vector[0], vector[1], vector[2]}};
          }

      std::vector<tangent::point_t> seeds;
      for (auto x = 0; x < slider_x->value(); x++)
        for (auto y = 0; y < slider_y->value(); y++)
          for (auto z = 0; z < slider_z->value(); z++)
            seeds.push_back({{spacing[0] * x, spacing[1] * y, spacing[2] * z, 0.0F}});

      tangent::TraceRecorder recorder;
      tangent::OmpCartGridStreamlineTracer tracer(&recorder);
      tracer.SetData              (&data);
      tracer.SetIntegrationStep   (float(slider_integration_step->value()) / slider_integration_step->maximum());
      tracer.SetNumberOfIterations(slider_iterations->value());
      auto output = tracer.TraceSeeds(seeds);

      auto& population = recorder.GetPopulation();
      for(auto i = 0; i < population.GetNumberOfTraces(); i++)
      {
        auto& path = population[i];
        for(auto j = 0; j < path.size() - 1; j++)
        {
          float3 start      = {path[j]    [0], -path[j]    [1], path[j]    [2]};
          float3 end        = {path[j + 1][0], -path[j + 1][1], path[j + 1][2]};
          auto   difference = fabs     (end - start);
          auto   normalized = normalize(difference );
          points.push_back(start);
          points.push_back(end  );
          for(auto k = 0; k < 2; ++k)
            colors.push_back(float4{normalized.x, normalized.z, normalized.y, 1.0});
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

  selector->setEnabled(true);
  button_trace_selection->setEnabled(true);
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  if (unit_vectors.is_initialized() && unit_vectors.get().num_elements() > 0)
  {
    logger_->info(std::string("Trace successful."));
    streamline_renderer_->set_data(points, colors);
  }
  else
  {
    logger_->info(std::string("Trace failed: Tractography only supports unit vectors (MSA-0309 style) at the moment."));
  }
}
}
