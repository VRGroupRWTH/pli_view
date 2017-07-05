#include /* implements */ <ui/plugins/volume_rendering_plugin.hpp>

#include <boost/format.hpp>

#include <third_party/qwt/qwt_plot_grid.h>
#include <third_party/qwt/qwt_plot_histogram.h>
#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>
#include <visualization/volume_renderer.hpp>

namespace pli
{
volume_rendering_plugin::volume_rendering_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);

  connect(checkbox_enabled   , &QCheckBox::stateChanged   , [&](bool state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    volume_renderer_->set_active(state);
  });
  connect(slider_step_size   , &QSlider::valueChanged     , [&]
  {
    auto step_size = float(slider_step_size->value()) / slider_step_size->maximum();
    line_edit_step_size->setText(QString::fromStdString((boost::format("%.4f") % step_size).str()));
  });
  connect(slider_step_size   , &QSlider::sliderReleased   , [&]
  {
    volume_renderer_->set_step_size(line_edit_utility::get_text<double>(line_edit_step_size));
  });
  connect(line_edit_step_size, &QLineEdit::editingFinished, [&]
  {
    auto step_size = line_edit_utility::get_text<double>(line_edit_step_size);
    slider_step_size->setValue(step_size * slider_step_size->maximum());
    volume_renderer_->set_step_size(step_size);
  });

  // Setup transfer function widget.
  auto font = transfer_function_widget->axisFont(0);
  font.setPointSize(8);
  transfer_function_widget->setAxisFont     (0, font);
  transfer_function_widget->setAxisFont     (2, font);
  transfer_function_widget->setAxisAutoScale(0);
  transfer_function_widget->setAxisScale    (2, 0, 255);
  transfer_function_widget->setAutoReplot   (true);

  auto grid  = new QwtPlotGrid;
  histogram_ = new QwtPlotHistogram;
  grid      ->enableXMin (true);
  grid      ->enableYMin (true);
  grid      ->setMajorPen(QPen  (Qt::black, 0, Qt::DotLine  ));
  grid      ->setMinorPen(QPen  (Qt::gray , 0, Qt::DotLine  ));
  histogram_->setPen     (QPen  (Qt::gray , 0));
  histogram_->setBrush   (QBrush(Qt::gray));
  histogram_->setStyle   (QwtPlotHistogram::HistogramStyle::Columns);
  grid      ->attach     (transfer_function_widget);
  histogram_->attach     (transfer_function_widget);
}
void volume_rendering_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>    (), &data_plugin::on_change    , [&]
  {
    upload();
  });
  connect(owner_->get_plugin<selector_plugin>(), &selector_plugin::on_change, [&]
  {
    upload();
  });
 
  auto step_size = double(slider_step_size->value()) / slider_step_size->maximum();

  volume_renderer_ = owner_->viewer->add_renderable<volume_renderer>();
  volume_renderer_->set_active   (checkbox_enabled->isChecked());
  volume_renderer_->set_step_size(step_size);
  
  line_edit_step_size->setText(QString::fromStdString((boost::format("%.4f") % step_size).str()));

  // TODO: CREATE TRANSFER FUNCTION EDITOR AND PROVIDE THIS FROM THERE.
  std::vector<float4> transfer_function(256, float4{0.0F, 0.0F, 0.0F, 0.0F});
  for (auto i = 125; i < 255; i++)
    transfer_function[i] = float4{float(i) / 255.0F, float(i) / 255.0F, float(i) / 255.0F, float(i) / 255.0F};
  volume_renderer_->set_transfer_function(transfer_function);

  logger_->info(std::string("Start successful."));
}
void volume_rendering_plugin::upload()
{
  logger_->info(std::string("Updating viewer..."));

  auto io       = owner_->get_plugin<pli::data_plugin>    ()->io();
  auto selector = owner_->get_plugin<pli::selector_plugin>();
  auto offset   = selector->selection_offset();
  auto size     = selector->selection_size  ();
  auto stride   = selector->selection_stride();

  if  (io == nullptr || size[0] == 0 || size[1] == 0 || size[2] == 0)
  {
    logger_->info(std::string("Update failed: No data."));
    return;
  }

  size = {size[0] / stride[0], size[1] / stride[1], size[2] / stride[2]};

  owner_->viewer->set_wait_spinner_enabled(true);
  selector      ->setEnabled(false);

  // Load data from hard drive (on another thread).
  std::array<float, 3>                          spacing    ;
  boost::optional<boost::multi_array<float, 3>> retardation;
  std::vector<std::size_t>                      quantified_retardation(255);
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      spacing = io->load_vector_spacing();
      retardation.reset(io->load_retardation_dataset(offset, size, stride, false));
      std::for_each(retardation->origin(), retardation->origin() + retardation->num_elements(), [&quantified_retardation] (const float& value) { quantified_retardation[value]++; });
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  });
  while(future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();
    
  // Update transfer function widget.
  QVector<QwtIntervalSample> samples;
  for (auto i = 0; i < 255; i++)
    samples.push_back(QwtIntervalSample(quantified_retardation[i], i, i + 1));
  histogram_->setSamples(samples);

  uint3  cuda_size   {unsigned(size[0]), unsigned(size[1]), unsigned(size[2])};
  float3 cuda_spacing{spacing[0], spacing[1], spacing[2]};
  volume_renderer_->set_data(cuda_size, cuda_spacing, retardation.get().data());

  selector      ->setEnabled(true);
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Update successful."));
}
}
