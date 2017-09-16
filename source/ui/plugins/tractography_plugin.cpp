#include <pli_vis/ui/plugins/tractography_plugin.hpp>

#include <algorithm>
#include <random>

#include <boost/format.hpp>
#include <tangent-base/default_tracers.hpp>

#include <pli_vis/cuda/pt/tracer.h>
#include <pli_vis/cuda/utility/vector_ops.h>
#include <pli_vis/ui/plugins/data_plugin.hpp>
#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
tractography_plugin::tractography_plugin(QWidget* parent) : plugin(parent)
{
  line_edit_integration_step->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));
  line_edit_iterations      ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
 
  line_edit_integration_step->setText(QString::fromStdString((boost::format("%.4f") % (float(slider_integration_step->value()) / slider_integration_step->maximum())).str()));
  line_edit_iterations      ->setText(QString::fromStdString(std::to_string(slider_iterations   ->value())));

  connect(checkbox_enabled          , &QCheckBox::stateChanged    , [&] (bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    streamline_renderer_->set_active(state);
  });
  connect(image                     , &roi_selector::on_selection_change, [&](const std::array<float, 2> offset_perc, const std::array<float, 2> size_perc)
  {
    std::array<int, 2> offset { int(offset_perc[0] * slider_x->maximum()), int(offset_perc[1] * slider_y->maximum()) };
    std::array<int, 2> size   { int(size_perc  [0] * slider_x->maximum()), int(size_perc  [1] * slider_y->maximum()) };
    line_edit_offset_x->setText      (QString::fromStdString(std::to_string(offset[0])));
    line_edit_size_x  ->setText      (QString::fromStdString(std::to_string(size  [0])));
    line_edit_offset_y->setText      (QString::fromStdString(std::to_string(offset[1])));
    line_edit_size_y  ->setText      (QString::fromStdString(std::to_string(size  [1])));
    slider_x          ->setLowerValue(offset[0]);
    slider_x          ->setUpperValue(offset[0] + size[0]);
    slider_y          ->setLowerValue(offset[1]);
    slider_y          ->setUpperValue(offset[1] + size[1]);
  });
  connect(slider_x                  , &QxtSpanSlider::lowerValueChanged , [&](int value)
  {
    line_edit_offset_x->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_x  ->setText(QString::fromStdString(std::to_string(slider_x->upperValue() - value)));
  });
  connect(slider_x                  , &QxtSpanSlider::upperValueChanged , [&](int value)
  {
    line_edit_size_x->setText(QString::fromStdString(std::to_string(value - slider_x->lowerValue())));
  });
  connect(slider_x                  , &QxtSpanSlider::sliderReleased    , [&]
  {
    image->set_selection_offset_percentage({static_cast<float>(slider_x->lowerValue())                          / slider_x->maximum(), image->selection_offset_percentage()[1]});
    image->set_selection_size_percentage  ({static_cast<float>(slider_x->upperValue() - slider_x->lowerValue()) / slider_x->maximum(), image->selection_size_percentage  ()[1]});
  });
  connect(slider_y                  , &QxtSpanSlider::lowerValueChanged , [&](int value)
  {
    line_edit_offset_y->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_y  ->setText(QString::fromStdString(std::to_string(slider_y->upperValue() - value)));
  });
  connect(slider_y                  , &QxtSpanSlider::upperValueChanged , [&](int value)
  {
    line_edit_size_y->setText(QString::fromStdString(std::to_string(value - slider_y->lowerValue())));
  });
  connect(slider_y                  , &QxtSpanSlider::sliderReleased    , [&]
  {
    image->set_selection_offset_percentage({image->selection_offset_percentage()[0], static_cast<float>(slider_y->lowerValue())                          / slider_y->maximum()});
    image->set_selection_size_percentage  ({image->selection_size_percentage  ()[0], static_cast<float>(slider_y->upperValue() - slider_y->lowerValue()) / slider_y->maximum()});
  });
  connect(slider_z                  , &QxtSpanSlider::lowerValueChanged , [&](int value)
  {
    line_edit_offset_z->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_z  ->setText(QString::fromStdString(std::to_string(slider_z->upperValue() - value)));
  });
  connect(slider_z                  , &QxtSpanSlider::upperValueChanged , [&](int value)
  {
    line_edit_size_z->setText(QString::fromStdString(std::to_string(value - slider_z->lowerValue())));
  });
  connect(line_edit_offset_x        , &QLineEdit::editingFinished       , [&]
  {
    auto value = std::max(std::min(line_edit::get_text<int>(line_edit_offset_x), int(slider_x->maximum())), int(slider_x->minimum()));
    if (slider_x->upperValue() < value)
      slider_x->setUpperValue(value);
    slider_x        ->setLowerValue                  (value);
    image           ->set_selection_offset_percentage({static_cast<float>(value) / slider_x->maximum(), image->selection_offset_percentage()[1]});
    image           ->set_selection_size_percentage  ({static_cast<float>(slider_x->upperValue() - slider_x->lowerValue()) / slider_x->maximum(), image->selection_size_percentage()[1]});
    line_edit_size_x->setText                        (QString::fromStdString(std::to_string(slider_x->upperValue() - value)));
  });
  connect(line_edit_size_x          , &QLineEdit::editingFinished       , [&]
  {
    auto value = std::max(std::min(line_edit::get_text<int>(line_edit_size_x), int(slider_x->maximum())), int(slider_x->minimum()));
    slider_x->setUpperValue                (slider_x->lowerValue() + value);
    image   ->set_selection_size_percentage({static_cast<float>(value) / slider_x->maximum(), image->selection_size_percentage()[1]});
  });
  connect(line_edit_offset_y        , &QLineEdit::editingFinished       , [&]
  {
    auto value = std::max(std::min(line_edit::get_text<int>(line_edit_offset_y), int(slider_y->maximum())), int(slider_y->minimum()));
    if (slider_y->upperValue() < value)
      slider_y->setUpperValue(value);
    slider_y        ->setLowerValue                  (value);
    image           ->set_selection_offset_percentage({image->selection_offset_percentage()[0], static_cast<float>(value) / slider_y->maximum()});
    image           ->set_selection_size_percentage  ({image->selection_size_percentage  ()[0], static_cast<float>(slider_y->upperValue() - slider_y->lowerValue()) / slider_y->maximum()});
    line_edit_size_y->setText                        (QString::fromStdString(std::to_string(slider_y->upperValue() - value)));
  });
  connect(line_edit_size_y          , &QLineEdit::editingFinished       , [&]
  {
    auto value = std::max(std::min(line_edit::get_text<int>(line_edit_size_y), int(slider_y->maximum())), int(slider_y->minimum()));
    slider_y->setUpperValue                (slider_y->lowerValue() + value);
    image   ->set_selection_size_percentage({image->selection_size_percentage()[0], static_cast<float>(value) / slider_y->maximum()});
  });
  connect(line_edit_offset_z        , &QLineEdit::editingFinished       , [&]
  {
    auto value = std::max(std::min(line_edit::get_text<int>(line_edit_offset_z), int(slider_z->maximum())), int(slider_z->minimum()));
    if (slider_z->upperValue() < value)
      slider_z->setUpperValue(value + 1);
    slider_z        ->setLowerValue(value);
    line_edit_size_z->setText      (QString::fromStdString(std::to_string(slider_z->upperValue() - value)));
  });
  connect(line_edit_size_z          , &QLineEdit::editingFinished       , [&]
  {
    auto value = std::min(line_edit::get_text<int>(line_edit_size_z), int(slider_z->maximum() - slider_z->minimum()));
    slider_z->setUpperValue(slider_z->lowerValue() + value);
  });
  connect(slider_integration_step   , &QSlider::valueChanged            , [&]
  {
    line_edit_integration_step->setText(QString::fromStdString((boost::format("%.4f") % (float(slider_integration_step->value()) / slider_integration_step->maximum())).str()));
  });
  connect(line_edit_integration_step, &QLineEdit::editingFinished       , [&]
  {
    slider_integration_step->setValue(line_edit::get_text<double>(line_edit_integration_step) * slider_integration_step->maximum());
  });
  connect(slider_iterations         , &QSlider::valueChanged            , [&]
  {
    line_edit_iterations->setText(QString::fromStdString(boost::lexical_cast<std::string>(slider_iterations->value())));
  });
  connect(line_edit_iterations      , &QLineEdit::editingFinished       , [&]
  {
    slider_iterations->setValue(line_edit::get_text<int>(line_edit_iterations));
  });
  connect(button_trace_selection    , &QPushButton::clicked             , [&]
  {
    trace();
  });

  connect(radio_button_cpu, &QRadioButton::clicked, [&]
  {
    logger_->info(std::string("CPU particle tracing selected."));
    gpu_tracing_ = false;
  });
  connect(radio_button_gpu, &QRadioButton::clicked, [&]
  {
    logger_->info(std::string("GPU particle tracing selected."));
    gpu_tracing_ = true;
  });
}

void tractography_plugin::start()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  streamline_renderer_ = owner_->viewer->add_renderable<streamline_renderer>();

  connect(owner_->get_plugin<pli::data_plugin>(), &data_plugin::on_load, [&]
  {
    auto data_plugin = owner_->get_plugin<pli::data_plugin>();

    // Adjust slider boundaries.
    auto size = data_plugin->selection_size  ();
    slider_x->setMinimum(0); slider_x->setMaximum(size[0]);
    slider_y->setMinimum(0); slider_y->setMaximum(size[1]);
    slider_z->setMinimum(0); slider_z->setMaximum(size[2]);

    // Generate selection image.
    auto selection_image = data_plugin->generate_selection_image();
    auto shape           = selection_image.shape();
    image->setPixmap(QPixmap::fromImage(QImage(selection_image.data(), shape[0], shape[1], QImage::Format::Format_Grayscale8)));

    // Adjust widget size.
    image    ->setSizeIncrement(shape[0], shape[1]);
    letterbox->setWidget(image);
    letterbox->update();
    update();
    
    // Hack for enforcing a UI update.
    auto sizes = owner_->splitter->sizes();
    owner_->splitter->setSizes(QList<int>{0       , sizes[1]});
    owner_->splitter->setSizes(QList<int>{sizes[0], sizes[1]});
    owner_->update();
  });
}
void tractography_plugin::trace()
{
  owner_->set_wait_spinner_enabled(true );
  owner_->toolbox->setEnabled     (false);

  logger_->info(std::string("Tracing..."));

  std::vector<float3> points        ;
  std::vector<float3> directions    ;
  std::vector<float4> random_vectors;
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      auto vectors = owner_->get_plugin<data_plugin>()->generate_vectors(true);
      auto shape   = vectors.shape();

      if(!gpu_tracing_)
      {
        tangent::CartesianGrid data(tangent::grid_dim_t{{shape[0], shape[1], shape[2]}}, tangent::vector_t{{1.0, 1.0, 1.0}});
        auto data_ptr = data.GetVectorPointer(0);
        for (auto x = 0; x < shape[0]; x++)
          for (auto y = 0; y < shape[1]; y++)
            for (auto z = 0; z < shape[2]; z++)
            {
              auto vector = vectors[x][y][z];
              data_ptr[x + shape[0] * (y + shape[1] * z)] = tangent::vector_t{{vector.x, vector.y, vector.z}};
            }

        auto offset = seed_offset();
        auto size   = seed_size  ();
        auto stride = seed_stride();
        std::vector<tangent::point_t> seeds;
        for (auto x = offset[0]; x < offset[0] + size[0]; x+= stride[0])
          for (auto y = offset[1]; y < offset[1] + size[1]; y += stride[1])
            for (auto z = offset[2]; z < offset[2] + size[2]; z += stride[2])
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
      else
      {
        std::vector<float3> data(shape[0] * shape[1] * shape[2]);
        for (auto x = 0; x < shape[0]; x++)
          for (auto y = 0; y < shape[1]; y++)
            for (auto z = 0; z < shape[2]; z++)
              data[x + shape[0] * (y + shape[1] * z)] = vectors[x][y][z];

        std::vector<float3> seeds;
        auto offset = seed_offset();
        auto size   = seed_size  ();
        auto stride = seed_stride();
        for (auto x = offset[0]; x < offset[0] + size[0]; x += stride[0])
          for (auto y = offset[1]; y < offset[1] + size[1]; y += stride[1])
            for (auto z = offset[2]; z < offset[2] + size[2]; z += stride[2])
              seeds.push_back(float3{float(x), float(y), float(z)});
        
        auto traces = cupt::trace(
          slider_iterations->value(),
          float(slider_integration_step->value()) / slider_integration_step->maximum(),
          uint3  {unsigned(shape[0]), unsigned(shape[1]), unsigned(shape[2])},
          float3 {1.0f, 1.0f, 1.0f},
          data ,
          seeds);

        for (auto i = 0; i < traces.size(); ++i)
          for (auto j = 0; j < slider_iterations->value() - 1; ++j)
          {
            auto& start = traces[i][j  ];
            auto& end   = traces[i][j+1];
            if(end.x != 0 && end.y != 0 && end.z != 0)
            {
              auto direction = normalize(fabs(end - start));
              points.push_back(start);
              points.push_back(end);
              for (auto k = 0; k < 2; ++k)
                directions.push_back(direction);
            }
          }
      }

      std::random_device                    random_device;
      std::mt19937                          mersenne_twister(random_device());
      std::uniform_real_distribution<float> distribution;
      random_vectors.resize(streamline_renderer_->screen_size()[0] * streamline_renderer_->screen_size()[1] * streamline_renderer_->ao_samples());
      std::generate(random_vectors.begin(), random_vectors.end(), [&mersenne_twister, &distribution] ()
      {
        return float4{
          distribution(mersenne_twister), 
          distribution(mersenne_twister), 
          distribution(mersenne_twister), 
          distribution(mersenne_twister)};
      });
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
    streamline_renderer_->set_data(points, directions, random_vectors);
  }
  else
  {
    logger_->info(std::string("Trace failed."));
  }

  owner_->toolbox->setEnabled     (true );
  owner_->set_wait_spinner_enabled(false);
}
  
std::array<std::size_t, 3> tractography_plugin::seed_offset() const
{
  return
  {
    line_edit::get_text<std::size_t>(line_edit_offset_x),
    line_edit::get_text<std::size_t>(line_edit_offset_y),
    line_edit::get_text<std::size_t>(line_edit_offset_z)
  };
}
std::array<std::size_t, 3> tractography_plugin::seed_size  () const
{
  return
  {
    line_edit::get_text<std::size_t>(line_edit_size_x),
    line_edit::get_text<std::size_t>(line_edit_size_y),
    line_edit::get_text<std::size_t>(line_edit_size_z)
  };
}
std::array<std::size_t, 3> tractography_plugin::seed_stride() const
{
  return
  {
    line_edit::get_text<std::size_t>(line_edit_stride_x),
    line_edit::get_text<std::size_t>(line_edit_stride_y),
    line_edit::get_text<std::size_t>(line_edit_stride_z)
  };
}
}
