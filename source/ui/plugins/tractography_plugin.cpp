#include <pli_vis/ui/plugins/tractography_plugin.hpp>

#include <boost/format.hpp>
#include <tangent-base/default_tracers.hpp>

#include <pli_vis/cuda/sh/convert.h>
#include <pli_vis/cuda/sh/vector_ops.h>
#include <pli_vis/ui/plugins/data_plugin.hpp>
#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
tractography_plugin::tractography_plugin(QWidget* parent) : plugin(parent)
{
  line_edit_x               ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_y               ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_z               ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
  line_edit_integration_step->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));
  line_edit_iterations      ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
 
  line_edit_integration_step->setText(QString::fromStdString((boost::format("%.4f") % (float(slider_integration_step->value()) / slider_integration_step->maximum())).str()));
  line_edit_iterations      ->setText(QString::fromStdString(std::to_string(slider_iterations->value())));
  line_edit_x               ->setText(QString::fromStdString(std::to_string(slider_x         ->value())));
  line_edit_y               ->setText(QString::fromStdString(std::to_string(slider_y         ->value())));
  line_edit_z               ->setText(QString::fromStdString(std::to_string(slider_z         ->value())));
  line_edit_rate_of_decay   ->setText(QString::fromStdString(std::to_string(slider_rate_of_decay->value())));

  connect(checkbox_enabled          , &QCheckBox::stateChanged    , [&] (bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
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
    line_edit_integration_step->setText(QString::fromStdString((boost::format("%.4f") % (float(slider_integration_step->value()) / slider_integration_step->maximum())).str()));
  });
  connect(line_edit_integration_step, &QLineEdit::editingFinished , [&]
  {
    slider_integration_step->setValue(line_edit::get_text<double>(line_edit_integration_step) * slider_integration_step->maximum());
  });
  connect(slider_iterations         , &QSlider::valueChanged      , [&]
  {
    line_edit_iterations->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_iterations->value())));
  });
  connect(line_edit_iterations      , &QLineEdit::editingFinished , [&]
  {
    slider_iterations->setValue(line_edit::get_text<int>(line_edit_iterations));
  });
  connect(checkbox_view_dependent   , &QCheckBox::stateChanged    , [&] (bool state)
  {
    logger_->info(std::string("View dependent transparency is " + state ? "enabled." : "disabled."));
    streamline_renderer_->set_view_dependent_transparency(state);
    label_rate_of_decay    ->setEnabled(state);
    slider_rate_of_decay   ->setEnabled(state);
    line_edit_rate_of_decay->setEnabled(state);
  });
  connect(slider_rate_of_decay      , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_rate_of_decay->setText(QString::fromStdString(std::to_string(slider_rate_of_decay->value())));
    streamline_renderer_->set_view_dependent_rate_of_decay(slider_rate_of_decay->value());
  });
  connect(line_edit_rate_of_decay   , &QLineEdit::editingFinished , [&]
  {
    auto value = line_edit::get_text<int>(line_edit_rate_of_decay);
    slider_rate_of_decay->setValue(value);
    streamline_renderer_->set_view_dependent_rate_of_decay(value);
  });
  connect(button_trace_selection    , &QPushButton::clicked       , [&]
  {
    trace();
  });
}

void tractography_plugin::start()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  streamline_renderer_ = owner_->viewer->add_renderable<streamline_renderer>();
}
void tractography_plugin::trace()
{
  owner_->viewer ->set_wait_spinner_enabled(true );
  owner_->toolbox->setEnabled              (false);

  logger_->info(std::string("Tracing..."));

  std::vector<float3> points    ;
  std::vector<float3> directions;
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      auto vectors = owner_->get_plugin<data_plugin>()->generate_vectors(true);
      auto shape   = vectors.shape();

      tangent::CartesianGrid data(tangent::grid_dim_t{{shape[0], shape[1], shape[2]}}, tangent::vector_t{{1.0, 1.0, 1.0}});
      auto data_ptr = data.GetVectorPointer(0);
      for (auto x = 0; x < shape[0]; x++)
        for (auto y = 0; y < shape[1]; y++)
          for (auto z = 0; z < shape[2]; z++)
          {
            auto vector = vectors[x][y][z];
            data_ptr[x + shape[0] * (y + shape[1] * z)] = tangent::vector_t{{vector.x, vector.y, vector.z}};
          }

      std::vector<tangent::point_t> seeds;
      for (auto x = 0; x < slider_x->value(); x++)
        for (auto y = 0; y < slider_y->value(); y++)
          for (auto z = 0; z < slider_z->value(); z++)
            seeds.push_back({{float(x), float(y), float(z), 0.0F}});

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
          float3 start {path[j]    [0], path[j]    [1], path[j]    [2]};
          float3 end   {path[j + 1][0], path[j + 1][1], path[j + 1][2]};
          auto   direction = normalize(fabs(end - start));
          points.push_back(start);
          points.push_back(end  );   
          for(auto k = 0; k < 2; ++k)
            directions.push_back(direction);
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

  if (points.size() > 0 && directions.size() > 0)
  {
    logger_->info(std::string("Trace successful."));
    streamline_renderer_->set_data(points, directions);
  }
  else
  {
    logger_->info(std::string("Trace failed."));
  }

  owner_->toolbox->setEnabled              (true );
  owner_->viewer ->set_wait_spinner_enabled(false);
}
}
