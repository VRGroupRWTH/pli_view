#include /* implements */ <ui/plugins/fdm_plugin.hpp>

#include <limits>
#include <string>

#include <boost/format.hpp>
#include <boost/optional.hpp>

#include <ui/plugins/data_plugin.hpp>
#include <ui/plugins/selector_plugin.hpp>
#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>
#include <visualization/odf_field.hpp>

namespace pli
{
fdm_plugin::fdm_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);

  line_edit_longitude->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_latitude ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  
  connect(checkbox_enabled, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    odf_field_->set_active(state);
  });
  connect(checkbox_depth_0, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info("Grid layer 0 is {}.", state ? "enabled" : "disabled");
    set_visible_layers();
  });
  connect(checkbox_depth_1, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info("Grid layer 1 is {}.", state ? "enabled" : "disabled");
    set_visible_layers();
  });
  connect(checkbox_depth_2, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info("Grid layer 2 is {}.", state ? "enabled" : "disabled");
    set_visible_layers();
  });
  connect(checkbox_depth_3, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info("Grid layer 3 is {}.", state ? "enabled" : "disabled");
    set_visible_layers();
  });
  connect(checkbox_depth_4, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info("Grid layer 4 is {}.", state ? "enabled" : "disabled");
    set_visible_layers();
  });
  connect(checkbox_depth_5, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info("Grid layer 5 is {}.", state ? "enabled" : "disabled");
    set_visible_layers();
  });
  connect(checkbox_depth_6, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info("Grid layer 6 is {}.", state ? "enabled" : "disabled");
    set_visible_layers();
  });
  connect(checkbox_depth_7, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info("Grid layer 7 is {}.", state ? "enabled" : "disabled");
    set_visible_layers();
  });
  connect(checkbox_depth_8, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info("Grid layer 8 is {}.", state ? "enabled" : "disabled");
    set_visible_layers();
  });
  connect(checkbox_depth_9, &QCheckBox::stateChanged, [&](int state)
  {
    logger_->info("Grid layer 9 is {}.", state ? "enabled" : "disabled");
    set_visible_layers();
  });

  connect(slider_longitude   , &QxtSpanSlider::valueChanged  , [&]
  {
    line_edit_longitude->setText(QString::fromStdString(std::to_string(slider_longitude->value())));
  });
  connect(slider_longitude   , &QxtSpanSlider::sliderReleased, [&]
  {
    upload();
  });
  connect(slider_latitude    , &QxtSpanSlider::valueChanged  , [&]
  {
    line_edit_latitude->setText(QString::fromStdString(std::to_string(slider_latitude->value())));
  });
  connect(slider_latitude    , &QxtSpanSlider::sliderReleased, [&]
  {
    upload();
  });
  connect(line_edit_longitude, &QLineEdit::editingFinished   , [&]
  {
    slider_longitude->setValue(line_edit_utility::get_text<std::size_t>(line_edit_longitude));
    upload();
  });
  connect(line_edit_latitude , &QLineEdit::editingFinished   , [&]
  {
    slider_latitude->setValue(line_edit_utility::get_text<std::size_t>(line_edit_latitude));
    upload();
  });

  connect(checkbox_clustering_enabled, &QCheckBox::stateChanged    , [&](int state)
  {
    logger_->info("Clustering is {}.", state ? "enabled" : "disabled");
    label_threshold    ->setEnabled(state);
    slider_threshold   ->setEnabled(state);
    line_edit_threshold->setEnabled(state);
    upload();
  });
  connect(slider_threshold           , &QxtSpanSlider::valueChanged, [&]
  {
    auto threshold = threshold_multiplier_ * slider_threshold->value();
    line_edit_threshold->setText(QString::fromStdString((boost::format("%.2f") % threshold).str()));
  });
  connect(slider_threshold           , &QSlider::sliderReleased    , [&]()
  {
    upload();
  });
  connect(line_edit_threshold        , &QLineEdit::editingFinished , [&]
  {
    auto threshold = line_edit_utility::get_text<double>(line_edit_threshold);
    slider_threshold->setValue(threshold / threshold_multiplier_);
    upload();
  });
}

void fdm_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  line_edit_longitude->setText(QString::fromStdString(std::to_string(slider_longitude->value())));
  line_edit_latitude ->setText(QString::fromStdString(std::to_string(slider_latitude ->value())));
  line_edit_threshold->setText(QString::fromStdString((boost::format("%.2f") % (threshold_multiplier_ * slider_threshold->value())).str()));

  connect(owner_->get_plugin<data_plugin>    (), &data_plugin    ::on_change, [&]
  {
    upload();
  });
  connect(owner_->get_plugin<selector_plugin>(), &selector_plugin::on_change, [&]
  {
    upload();
  });

  odf_field_ = owner_->viewer->add_renderable<odf_field>();

  set_visible_layers();
 
  logger_->info(std::string("Start successful."));
}
void fdm_plugin::upload()
{
  logger_->info(std::string("Updating viewer..."));

  auto io       = owner_->get_plugin<pli::data_plugin>    ()->io();
  auto selector = owner_->get_plugin<pli::selector_plugin>();
  auto offset   = selector->offset();
  auto size     = selector->size  ();
  uint2 tessellations =
  {line_edit_utility::get_text<std::size_t>(line_edit_longitude),
   line_edit_utility::get_text<std::size_t>(line_edit_latitude )};

  if  (io == nullptr || size[0] == 0 || size[1] == 0 || size[2] == 0)
  {
    logger_->info(std::string("Update failed: No data."));
    return;
  }

  owner_->viewer->set_wait_spinner_enabled(true);
  selector->setEnabled(false);

  // Load data from hard drive (on another thread).
  std::array<float      , 3>                    spacing      ;
  std::array<std::size_t, 3>                    block_size   ;
  boost::optional<boost::multi_array<float, 4>> distributions;
  future_ = std::async(std::launch::async, [&]()
  {
    try
    {
      spacing    = io->load_vector_spacing();
      block_size = io->load_block_size    ();
      distributions.reset(io->load_fiber_distribution_dataset(offset, size, {1, 1, 1}, false));

      // Roll dimensions to power of two.
      size[0] = pow(2, ceil(log(size[0]) / log(2)));
      size[1] = pow(2, ceil(log(size[1]) / log(2)));
      size[2] = pow(2, ceil(log(size[2]) / log(2)));
      (*distributions).resize(boost::extents[size[0]][size[1]][size[2]][(*distributions).shape()[3]]);
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  });
  while (future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();

  // Upload data to GPU.
  uint3  cuda_size      {unsigned(size[0]), unsigned(size[1]), unsigned(size[2])};
  float3 cuda_spacing   {spacing   [0], spacing   [1], spacing   [2]};
  uint3  cuda_block_size{block_size[0], block_size[1], block_size[2]};
  if (distributions.is_initialized() && distributions.get().num_elements() > 0)
    odf_field_->set_data(
      cuda_size, 
      distributions.get().shape()[3],
      distributions.get().data(), 
      tessellations, 
      cuda_spacing,
      cuda_block_size, 
      1.0, 
      checkbox_clustering_enabled->isChecked(),
      threshold_multiplier_ * float(slider_threshold->value()),
      [&] (const std::string& message) { logger_->info(message); });

  selector->setEnabled(true);
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Update successful."));
}

void fdm_plugin::set_visible_layers() const
{
  odf_field_->set_visible_layers({
    checkbox_depth_0->isChecked(),
    checkbox_depth_1->isChecked(),
    checkbox_depth_2->isChecked(),
    checkbox_depth_3->isChecked(),
    checkbox_depth_4->isChecked(),
    checkbox_depth_5->isChecked(),
    checkbox_depth_6->isChecked(),
    checkbox_depth_7->isChecked(),
    checkbox_depth_8->isChecked(),
    checkbox_depth_9->isChecked()
  });
  owner_->viewer->update();
}
}
