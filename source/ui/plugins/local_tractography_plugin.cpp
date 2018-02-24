#include <pli_vis/ui/plugins/local_tractography_plugin.hpp>

#include <algorithm>
#include <random>

#include <boost/format.hpp>
#include <QFileDialog>
#include <tangent-base/default_tracers.hpp>
#include <tangent-tbb/tbb_default_tracers.hpp>

#include <pli_vis/cuda/pt/tracer.h>
#include <pli_vis/cuda/utility/vector_ops.h>
#include <pli_vis/ui/plugins/data_plugin.hpp>
#include <pli_vis/ui/widgets/remote_viewer.hpp>
#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>
#include <pli_vis/utility/make_even.hpp>
#include <pli_vis/visualization/algorithms/streamline_renderer.hpp>
#include <pli_vis/visualization/algorithms/lineao_streamline_renderer.hpp>
#include <pli_vis/visualization/algorithms/ospray_streamline_exporter.hpp>
#include <pli_vis/visualization/algorithms/ospray_streamline_renderer.hpp>

namespace pli
{
local_tractography_plugin::local_tractography_plugin(QWidget* parent) : plugin(parent)
{
  line_edit_integration_step->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));
  line_edit_iterations      ->setValidator(new QIntValidator   (0, std::numeric_limits<int>   ::max(),     this));
 
  line_edit_integration_step->setText(QString::fromStdString((boost::format("%.4f") % (float(slider_integration_step->value()) / slider_integration_step->maximum())).str()));
  line_edit_iterations      ->setText(QString::fromStdString(std::to_string(slider_iterations   ->value())));

  connect(checkbox_enabled          , &QCheckBox::stateChanged          , [&] (bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    streamline_renderer_->set_active(state);
  });
  connect(image                     , &roi_selector::on_selection_change, [&] (const std::array<float, 2> offset_perc, const std::array<float, 2> size_perc)
  {
    std::array<int, 2> offset { make_even(int(offset_perc[0] * slider_x->maximum())), make_even(int(offset_perc[1] * slider_y->maximum())) };
    std::array<int, 2> size   { make_even(int(size_perc  [0] * slider_x->maximum())), make_even(int(size_perc  [1] * slider_y->maximum())) };
    line_edit_offset_x->setText      (QString::fromStdString(std::to_string(offset[0])));
    line_edit_size_x  ->setText      (QString::fromStdString(std::to_string(size  [0])));
    line_edit_offset_y->setText      (QString::fromStdString(std::to_string(offset[1])));
    line_edit_size_y  ->setText      (QString::fromStdString(std::to_string(size  [1])));
    slider_x          ->setLowerValue(offset[0]);
    slider_x          ->setUpperValue(offset[0] + size[0]);
    slider_y          ->setLowerValue(offset[1]);
    slider_y          ->setUpperValue(offset[1] + size[1]);
  });
  connect(slider_x                  , &QxtSpanSlider::lowerValueChanged , [&] (int value)
  {
    line_edit_offset_x->setText(QString::fromStdString(std::to_string(make_even(value))));
    line_edit_size_x  ->setText(QString::fromStdString(std::to_string(make_even(slider_x->upperValue() - value))));
  });
  connect(slider_x                  , &QxtSpanSlider::upperValueChanged , [&] (int value)
  {
    line_edit_size_x->setText(QString::fromStdString(std::to_string(make_even(value - slider_x->lowerValue()))));
  });
  connect(slider_x                  , &QxtSpanSlider::sliderReleased    , [&] 
  {
    image->set_selection_offset_percentage({static_cast<float>(slider_x->lowerValue())                          / slider_x->maximum(), image->selection_offset_percentage()[1]});
    image->set_selection_size_percentage  ({static_cast<float>(slider_x->upperValue() - slider_x->lowerValue()) / slider_x->maximum(), image->selection_size_percentage  ()[1]});
  });
  connect(slider_y                  , &QxtSpanSlider::lowerValueChanged , [&] (int value)
  {
    line_edit_offset_y->setText(QString::fromStdString(std::to_string(make_even(value))));
    line_edit_size_y  ->setText(QString::fromStdString(std::to_string(make_even(slider_y->upperValue() - value))));
  });
  connect(slider_y                  , &QxtSpanSlider::upperValueChanged , [&] (int value)
  {
    line_edit_size_y->setText(QString::fromStdString(std::to_string(make_even(value - slider_y->lowerValue()))));
  });
  connect(slider_y                  , &QxtSpanSlider::sliderReleased    , [&] 
  {
    image->set_selection_offset_percentage({image->selection_offset_percentage()[0], static_cast<float>(slider_y->lowerValue())                          / slider_y->maximum()});
    image->set_selection_size_percentage  ({image->selection_size_percentage  ()[0], static_cast<float>(slider_y->upperValue() - slider_y->lowerValue()) / slider_y->maximum()});
  });
  connect(slider_z                  , &QxtSpanSlider::lowerValueChanged , [&] (int value)
  {
    line_edit_offset_z->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_z  ->setText(QString::fromStdString(std::to_string(slider_z->upperValue() - value)));
  });
  connect(slider_z                  , &QxtSpanSlider::upperValueChanged , [&] (int value)
  {
    line_edit_size_z->setText(QString::fromStdString(std::to_string(value - slider_z->lowerValue())));
  });
  connect(line_edit_offset_x        , &QLineEdit::editingFinished       , [&] 
  {
    auto value = std::max(std::min(make_even(line_edit::get_text<int>(line_edit_offset_x)), int(slider_x->maximum())), int(slider_x->minimum()));
    if (slider_x->upperValue() < value)
      slider_x->setUpperValue(value);
    slider_x        ->setLowerValue                  (value);
    image           ->set_selection_offset_percentage({static_cast<float>(value) / slider_x->maximum(), image->selection_offset_percentage()[1]});
    image           ->set_selection_size_percentage  ({static_cast<float>(slider_x->upperValue() - slider_x->lowerValue()) / slider_x->maximum(), image->selection_size_percentage()[1]});
    line_edit_size_x->setText                        (QString::fromStdString(std::to_string(make_even(slider_x->upperValue() - value))));
  });
  connect(line_edit_size_x          , &QLineEdit::editingFinished       , [&] 
  {
    auto value = std::max(std::min(make_even(line_edit::get_text<int>(line_edit_size_x)), int(slider_x->maximum())), int(slider_x->minimum()));
    slider_x->setUpperValue                (make_even(slider_x->lowerValue() + value));
    image   ->set_selection_size_percentage({static_cast<float>(value) / slider_x->maximum(), image->selection_size_percentage()[1]});
  });
  connect(line_edit_offset_y        , &QLineEdit::editingFinished       , [&] 
  {
    auto value = std::max(std::min(make_even(line_edit::get_text<int>(line_edit_offset_y)), int(slider_y->maximum())), int(slider_y->minimum()));
    if (slider_y->upperValue() < value)
      slider_y->setUpperValue(value);
    slider_y        ->setLowerValue                  (value);
    image           ->set_selection_offset_percentage({image->selection_offset_percentage()[0], static_cast<float>(value) / slider_y->maximum()});
    image           ->set_selection_size_percentage  ({image->selection_size_percentage  ()[0], static_cast<float>(slider_y->upperValue() - slider_y->lowerValue()) / slider_y->maximum()});
    line_edit_size_y->setText                        (QString::fromStdString(std::to_string(make_even(slider_y->upperValue() - value))));
  });
  connect(line_edit_size_y          , &QLineEdit::editingFinished       , [&] 
  {
    auto value = std::max(std::min(make_even(line_edit::get_text<int>(line_edit_size_y)), int(slider_y->maximum())), int(slider_y->minimum()));
    slider_y->setUpperValue                (make_even(slider_y->lowerValue() + value));
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
  connect(button_local_trace        , &QPushButton::clicked             , [&]
  {
    trace();
  });
  connect(button_remote_trace       , &QPushButton::clicked             , [&]
  {
    remote_trace();
  });
  connect(button_ospray_export      , &QPushButton::clicked             , [&]
  {
    auto filepath = QFileDialog::getSaveFileName(this, tr("Select output file."), "C:/", tr("Image Files (*.bmp *.jpg *.png *.tga)")).toStdString();
    if (filepath.empty())
      return;

    auto camera   = owner_->viewer->camera();
    auto position = float3{camera->translation()[0], camera->translation()[1], camera->translation()[2]};
    auto forward  = float3{camera->forward    ()[0], camera->forward    ()[1], camera->forward    ()[2]};
    auto up       = float3{camera->up         ()[0], camera->up         ()[1], camera->up         ()[2]};
    auto size     = uint2 {unsigned(owner_->viewer->size().width()), unsigned(owner_->viewer->size().height())};
    ospray_streamline_exporter::to_image(position, forward, up, size, vertices_, tangents_, indices_, filepath);
  });
  connect(radio_button_cpu          , &QRadioButton::clicked            , [&]
  {
    logger_->info(std::string("CPU particle tracing selected."));
    gpu_tracing_ = false;
  });
  connect(radio_button_gpu          , &QRadioButton::clicked            , [&]
  {
    logger_->info(std::string("GPU particle tracing selected."));
    gpu_tracing_ = true;
  });
  connect(radio_button_regular      , &QRadioButton::clicked            , [&]
  {
    auto color_plugin = owner_->get_plugin<pli::color_plugin>();
    
    owner_->viewer->remove_renderable(streamline_renderer_);
    streamline_renderer_ = owner_->viewer->add_renderable<streamline_renderer>       ();
    streamline_renderer_->set_active(checkbox_enabled->isChecked());
    streamline_renderer_->set_color_mapping(color_plugin->mode(), color_plugin->k(), color_plugin->inverted());
  });
  connect(radio_button_lineao       , &QRadioButton::clicked            , [&]
  {
    auto color_plugin = owner_->get_plugin<pli::color_plugin>();

    owner_->viewer->remove_renderable(streamline_renderer_);
    streamline_renderer_ = owner_->viewer->add_renderable<lineao_streamline_renderer>();
    streamline_renderer_->set_active(checkbox_enabled->isChecked());
    streamline_renderer_->set_color_mapping(color_plugin->mode(), color_plugin->k(), color_plugin->inverted());
  });
  connect(radio_button_ospray       , &QRadioButton::clicked            , [&]
  {
    auto color_plugin = owner_->get_plugin<pli::color_plugin>();

    owner_->viewer->remove_renderable(streamline_renderer_);
    streamline_renderer_ = owner_->viewer->add_renderable<ospray_streamline_renderer>();
    streamline_renderer_->set_active(checkbox_enabled->isChecked());
    streamline_renderer_->set_color_mapping(color_plugin->mode(), color_plugin->k(), color_plugin->inverted());
  });
}

void local_tractography_plugin::start()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  if      (radio_button_regular->isChecked())
    streamline_renderer_ = owner_->viewer->add_renderable<streamline_renderer>       ();
  else if (radio_button_lineao ->isChecked())
    streamline_renderer_ = owner_->viewer->add_renderable<lineao_streamline_renderer>();
  else
    streamline_renderer_ = owner_->viewer->add_renderable<ospray_streamline_renderer>();

  connect(owner_->toolbox, &QToolBox::currentChanged, [&](int tab)
  {
    // Hack for enforcing a UI update.
    auto sizes = owner_->splitter->sizes();
    owner_->splitter->setSizes(QList<int>{0, sizes[1]});
    owner_->splitter->setSizes(QList<int>{sizes[0], sizes[1]});
    owner_->update();
    update();
  });
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
    update();
  });
  connect(owner_->get_plugin<color_plugin>(), &color_plugin::on_change, [&](int mode, float k, bool inverted)
  {
    streamline_renderer_->set_color_mapping(mode, k, inverted);
  });
}
void local_tractography_plugin::trace()
{
  owner_->set_is_loading(true);

  logger_->info(std::string("Tracing..."));

  vertices_.clear();
  tangents_.clear();
  indices_ .clear();

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
              auto& vector = vectors[x][y][z];
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
        tangent::TBBCartGridStreamlineTracer tracer(&recorder);
        tracer.SetData              (&data);
        tracer.SetIntegrationStep   (float(slider_integration_step->value()) / slider_integration_step->maximum());
        tracer.SetNumberOfIterations(slider_iterations->value());
        auto output = tracer.TraceSeeds(seeds);

        auto& population = recorder.GetPopulation();
        for (auto i = 0; i < population.GetNumberOfTraces(); ++i)
        {
          auto& path          = population[i];
          auto  vertex_offset = vertices_.size();
          for(auto j = 0; j < path.size(); ++j)
          {
            auto& vertex = path[j];
            if (isnan(vertex[0])  || 
                isnan(vertex[1])  || 
                isnan(vertex[2])  || 
                vertex[0] == 0.0F && 
                vertex[1] == 0.0F && 
                vertex[2] == 0.0F)
              continue;

            vertices_.push_back(float4{vertex[0], vertex[1], vertex[2], 1.0});
            
            if (j > 0)
              tangents_.push_back(normalize(float4{
                path[j][0] - path[j - 1][0], 
                path[j][1] - path[j - 1][1], 
                path[j][2] - path[j - 1][2], 
                0.0}));
            else if (path.size() > 1)
              tangents_.push_back(normalize(float4{
                path[j + 1][0] - path[j][0], 
                path[j + 1][1] - path[j][1], 
                path[j + 1][2] - path[j][2], 
                0.0}));
            else
              tangents_.push_back(float4{0.0F, 0.0F, 0.0F, 0.0F});

            if (j > 0)
            {
              indices_.push_back(vertex_offset + j - 1);
              indices_.push_back(vertex_offset + j);
            }
          }
        }
      }
      else
      {
        std::vector<float4> data(shape[0] * shape[1] * shape[2]);
        for (auto x = 0; x < shape[0]; x++)
          for (auto y = 0; y < shape[1]; y++)
            for (auto z = 0; z < shape[2]; z++)
            {
              auto& vector = vectors[x][y][z];
              data[x + shape[0] * (y + shape[1] * z)] = float4 {vector.x, vector.y, vector.z, 1.0};
            }

        auto offset = seed_offset();
        auto size   = seed_size  ();
        auto stride = seed_stride();
        std::vector<float4> seeds;
        for (auto x = offset[0]; x < offset[0] + size[0]; x += stride[0])
          for (auto y = offset[1]; y < offset[1] + size[1]; y += stride[1])
            for (auto z = offset[2]; z < offset[2] + size[2]; z += stride[2])
              seeds.push_back(float4{float(x), float(y), float(z), 0.0});
        
        auto traces = cupt::trace(
          slider_iterations->value(),
          float(slider_integration_step->value()) / slider_integration_step->maximum(),
          uint3  {unsigned(shape[0]), unsigned(shape[1]), unsigned(shape[2])},
          float4 {1.0f, 1.0f, 1.0f, 0.0f},
          data ,
          seeds);

        for (auto i = 0; i < traces.size(); ++i)
        {
          auto& path          = traces[i];
          auto  vertex_offset = vertices_.size();
          for(auto j = 0; j < path.size(); ++j)
          {
            auto& vertex = path[j];
            if (isnan(vertex.x)  || 
                isnan(vertex.y)  || 
                isnan(vertex.z)  || 
                vertex.x == 0.0F && 
                vertex.y == 0.0F && 
                vertex.z == 0.0F)
              continue;

            vertices_.push_back(float4{vertex.x, vertex.y, vertex.z, 1.0});
            
            if (j > 0)
              tangents_.push_back(normalize(float4{
                path[j].x - path[j - 1].x, 
                path[j].y - path[j - 1].y, 
                path[j].z - path[j - 1].z, 
                0.0}));
            else if (path.size() > 1)
              tangents_.push_back(normalize(float4{
                path[j + 1].x - path[j].x, 
                path[j + 1].y - path[j].y, 
                path[j + 1].z - path[j].z, 
                0.0}));
            else
              tangents_.push_back(float4{0.0F, 0.0F, 0.0F, 0.0F});

            if (j > 0)
            {
              indices_.push_back(vertex_offset + j - 1);
              indices_.push_back(vertex_offset + j);
            }
          }
        }
      }

      std::random_device                    random_device;
      std::mt19937                          mersenne_twister(random_device());
      std::uniform_real_distribution<float> distribution;

      auto lineao_renderer = dynamic_cast<lineao_streamline_renderer*>(streamline_renderer_);
      if  (lineao_renderer)
      {
        auto screen_size = lineao_renderer->screen_size();
        random_vectors.resize(screen_size[0] * screen_size[1] * lineao_renderer->ao_samples());
        std::generate(random_vectors.begin(), random_vectors.end(), [&mersenne_twister, &distribution]()
        {
          return float4{
            distribution(mersenne_twister),
            distribution(mersenne_twister),
            distribution(mersenne_twister),
            distribution(mersenne_twister) };
        });
      }
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  });
  while (future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();

  if (vertices_.size() > 0 && tangents_.size() > 0)
  {
    logger_->info(std::string("Trace successful."));

    auto regular_renderer = dynamic_cast<streamline_renderer*>       (streamline_renderer_);
    if  (regular_renderer) regular_renderer->set_data(vertices_, tangents_, indices_);
    
    //auto lineao_renderer  = dynamic_cast<lineao_streamline_renderer*>(streamline_renderer_);
    //if  (lineao_renderer)  lineao_renderer ->set_data(vertices_, tangents_, random_vectors);
    
    //auto ospray_renderer  = dynamic_cast<ospray_streamline_renderer*>(streamline_renderer_);
    //if  (ospray_renderer)  ospray_renderer ->set_data(vertices_, tangents_);
  }
  else
  {
    logger_->info(std::string("Trace failed."));
  }

  owner_->set_is_loading(false);
}
void local_tractography_plugin::remote_trace()
{
  remote_viewer_ = std::make_unique<remote_viewer>();
  remote_viewer_->on_close.connect([&]()
  {
    remote_viewer_.reset();
  });
}
  
std::array<std::size_t, 3> local_tractography_plugin::seed_offset() const
{
  return
  {
    line_edit::get_text<std::size_t>(line_edit_offset_x),
    line_edit::get_text<std::size_t>(line_edit_offset_y),
    line_edit::get_text<std::size_t>(line_edit_offset_z)
  };
}
std::array<std::size_t, 3> local_tractography_plugin::seed_size  () const
{
  return
  {
    line_edit::get_text<std::size_t>(line_edit_size_x),
    line_edit::get_text<std::size_t>(line_edit_size_y),
    line_edit::get_text<std::size_t>(line_edit_size_z)
  };
}
std::array<std::size_t, 3> local_tractography_plugin::seed_stride() const
{
  return
  {
    line_edit::get_text<std::size_t>(line_edit_stride_x),
    line_edit::get_text<std::size_t>(line_edit_stride_y),
    line_edit::get_text<std::size_t>(line_edit_stride_z)
  };
}
}
