#include /* implements */ <ui/plugins/fdm_plugin.hpp>

#include <limits>
#include <string>

#include <boost/format.hpp>
#include <boost/optional.hpp>

#include <cuda/odf_field.h>
#include <sh/spherical_harmonics.h>
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
  
  line_edit_vector_block_x   ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_vector_block_x   ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_vector_block_x   ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_histogram_theta  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_histogram_phi    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_maximum_sh_degree->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_sampling_theta   ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_sampling_phi     ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  
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

  connect(slider_vector_block_x      , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_vector_block_x->setText(QString::fromStdString(std::to_string(slider_vector_block_x->value())));
  });
  connect(line_edit_vector_block_x   , &QLineEdit::editingFinished , [&]
  {
    slider_vector_block_x->setValue(line_edit_utility::get_text<int>(line_edit_vector_block_x));
  });
  connect(slider_vector_block_y      , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_vector_block_y->setText(QString::fromStdString(std::to_string(slider_vector_block_y->value())));
  });
  connect(line_edit_vector_block_y   , &QLineEdit::editingFinished , [&]
  {
    slider_vector_block_y->setValue(line_edit_utility::get_text<int>(line_edit_vector_block_y));
  });
  connect(slider_vector_block_z      , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_vector_block_z->setText(QString::fromStdString(std::to_string(slider_vector_block_z->value())));
  });
  connect(line_edit_vector_block_z   , &QLineEdit::editingFinished , [&]
  {
    slider_vector_block_z->setValue(line_edit_utility::get_text<int>(line_edit_vector_block_z));
  });
  connect(slider_histogram_theta     , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_histogram_theta->setText(QString::fromStdString(std::to_string(slider_histogram_theta->value())));
  });
  connect(line_edit_histogram_theta  , &QLineEdit::editingFinished , [&]
  {
    slider_histogram_theta->setValue(line_edit_utility::get_text<int>(line_edit_histogram_theta));
  });
  connect(slider_histogram_phi       , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_histogram_phi->setText(QString::fromStdString(std::to_string(slider_histogram_phi->value())));
  });
  connect(line_edit_histogram_phi    , &QLineEdit::editingFinished , [&]
  {
    slider_histogram_phi->setValue(line_edit_utility::get_text<int>(line_edit_histogram_phi));
  });
  connect(slider_maximum_sh_degree   , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_maximum_sh_degree->setText(QString::fromStdString(std::to_string(slider_maximum_sh_degree->value())));
  });
  connect(line_edit_maximum_sh_degree, &QLineEdit::editingFinished , [&]
  {
    slider_maximum_sh_degree->setValue(line_edit_utility::get_text<int>(line_edit_maximum_sh_degree));
  });
  connect(slider_sampling_theta      , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_sampling_theta->setText(QString::fromStdString(std::to_string(slider_sampling_theta->value())));
  });
  connect(line_edit_sampling_theta   , &QLineEdit::editingFinished , [&]
  {
    slider_sampling_theta->setValue(line_edit_utility::get_text<int>(line_edit_sampling_theta));
  });
  connect(slider_sampling_phi        , &QxtSpanSlider::valueChanged, [&]
  {
    line_edit_sampling_phi->setText(QString::fromStdString(std::to_string(slider_sampling_phi->value())));
  });
  connect(line_edit_sampling_phi     , &QLineEdit::editingFinished , [&]
  {
    slider_sampling_phi->setValue(line_edit_utility::get_text<int>(line_edit_sampling_phi));
  });

  connect(checkbox_clustering_enabled, &QCheckBox::stateChanged    , [&](int state)
  {
    logger_->info("Clustering is {}.", state ? "enabled" : "disabled");
    label_threshold    ->setEnabled(state);
    slider_threshold   ->setEnabled(state);
    line_edit_threshold->setEnabled(state);
  });
  connect(slider_threshold           , &QxtSpanSlider::valueChanged, [&]
  {
    auto threshold = threshold_multiplier_ * slider_threshold->value();
    line_edit_threshold->setText(QString::fromStdString((boost::format("%.2f") % threshold).str()));
  });
  connect(line_edit_threshold        , &QLineEdit::editingFinished , [&]
  {
    auto threshold = line_edit_utility::get_text<double>(line_edit_threshold);
    slider_threshold->setValue(threshold / threshold_multiplier_);
  });

  connect(button_calculate           , &QAbstractButton::clicked   , [&]
  {
    calculate();
  });
  connect(button_extract_peaks       , &QAbstractButton::clicked   , [&]
  {
    extract_peaks();
  });
}

void fdm_plugin::start    ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  line_edit_vector_block_x   ->setText(QString::fromStdString(std::to_string(slider_vector_block_x   ->value())));
  line_edit_vector_block_y   ->setText(QString::fromStdString(std::to_string(slider_vector_block_y   ->value())));
  line_edit_vector_block_z   ->setText(QString::fromStdString(std::to_string(slider_vector_block_z   ->value())));
  line_edit_histogram_theta  ->setText(QString::fromStdString(std::to_string(slider_histogram_theta  ->value())));
  line_edit_histogram_phi    ->setText(QString::fromStdString(std::to_string(slider_histogram_phi    ->value())));
  line_edit_maximum_sh_degree->setText(QString::fromStdString(std::to_string(slider_maximum_sh_degree->value())));
  line_edit_sampling_theta   ->setText(QString::fromStdString(std::to_string(slider_sampling_theta   ->value())));
  line_edit_sampling_phi     ->setText(QString::fromStdString(std::to_string(slider_sampling_phi     ->value())));
  line_edit_threshold        ->setText(QString::fromStdString((boost::format("%.2f") % (threshold_multiplier_ * slider_threshold->value())).str()));

  odf_field_ = owner_->viewer->add_renderable<odf_field>();
  set_visible_layers();

  logger_->info(std::string("Resetting GPU."));
  cudaDeviceReset();

  logger_->info(std::string("Initializing cusolver and cublas."));
  cusolverDnCreate(&cusolver_);
  cublasCreate    (&cublas_  );

  logger_->info(std::string("Start successful."));
}
void fdm_plugin::destroy()
{
  logger_->info(std::string("Destroying cusolver and cublas."));
  cusolverDnDestroy(cusolver_);
  cublasDestroy    (cublas_  );
}

void fdm_plugin::calculate    ()
{
  logger_->info(std::string("Updating viewer..."));
  
  auto io         = owner_->get_plugin<pli::data_plugin>    ()->io();
  auto selector   = owner_->get_plugin<pli::selector_plugin>();
  auto offset     = selector->selection_offset();
  auto size       = selector->selection_size  ();
  auto stride     = selector->selection_stride();
  auto max_degree = line_edit_utility::get_text<unsigned>(line_edit_maximum_sh_degree);
  uint3 block_dimensions       = {
    line_edit_utility::get_text<unsigned>(line_edit_vector_block_x ), 
    line_edit_utility::get_text<unsigned>(line_edit_vector_block_y ), 
    line_edit_utility::get_text<unsigned>(line_edit_vector_block_z )};
  uint2 histogram_dimensions   = {
    line_edit_utility::get_text<unsigned>(line_edit_histogram_theta),
    line_edit_utility::get_text<unsigned>(line_edit_histogram_phi  )};
  uint2 sampling_dimensions    = { 
    line_edit_utility::get_text<unsigned>(line_edit_sampling_theta),
    line_edit_utility::get_text<unsigned>(line_edit_sampling_phi  )};

  if(io == nullptr || size[0] == 0 || size[1] == 0 || size[2] == 0)
  {
    logger_->info(std::string("Update failed: No data."));
    return;
  }

  size = {size[0] / stride[0], size[1] / stride[1], size[2] / stride[2]};
  uint3 coefficient_dimensions = {
    unsigned(size[0]) / block_dimensions.x, 
    unsigned(size[1]) / block_dimensions.y, 
    unsigned(size[2]) / block_dimensions.z };

  owner_->viewer->set_wait_spinner_enabled(true);
  button_calculate->setEnabled(false);
  selector        ->setEnabled(false);

  // Load data from hard drive (on another thread).
  std::array<float, 3>                          spacing    ;
  boost::optional<boost::multi_array<float, 3>> direction  ;
  boost::optional<boost::multi_array<float, 3>> inclination;
  boost::multi_array<float, 4> coefficients(boost::extents
    [coefficient_dimensions.x]
    [coefficient_dimensions.y]
    [coefficient_dimensions.z]
    [cush::coefficient_count(max_degree)]);
  future_ = std::async(std::launch::async, [&]
  {
    try
    {
      spacing = io->load_vector_spacing();
      direction  .reset(io->load_fiber_direction_dataset  (offset, size, stride, false));
      inclination.reset(io->load_fiber_inclination_dataset(offset, size, stride, false));
      
      calculate_odfs(
        cublas_,
        cusolver_,
        coefficient_dimensions,
        block_dimensions,
        histogram_dimensions,
        max_degree,
        direction   .get().data(),
        inclination .get().data(),
        coefficients.data(),
        [&](const std::string& message) { logger_->info(message); });
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  });
  while (future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();
  
  // Upload data to GPU.
  uint3  cuda_size    {unsigned(coefficient_dimensions.x), unsigned(coefficient_dimensions.y), unsigned(coefficient_dimensions.z)};
  float3 cuda_spacing {spacing[0], spacing[1], spacing[2]};
  if (coefficients.num_elements() > 0)
    odf_field_->set_data(
      cuda_size, 
      unsigned(coefficients.shape()[3]),
      coefficients.data (),
      sampling_dimensions, 
      cuda_spacing,
      block_dimensions, 
      1.0F, 
      checkbox_clustering_enabled->isChecked(),
      threshold_multiplier_ * float(slider_threshold->value()),
      [&] (const std::string& message) { logger_->info(message); });
  
  selector        ->setEnabled(true);
  button_calculate->setEnabled(true);
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Update successful."));
}
void fdm_plugin::extract_peaks()
{
  logger_->info(std::string("Extracting peaks..."));

  auto selector = owner_->get_plugin<pli::selector_plugin>();

  owner_->viewer->set_wait_spinner_enabled(true);
  button_calculate->setEnabled(false);
  selector        ->setEnabled(false);

  // TODO: Apply peak extraction.
  
  selector        ->setEnabled(true);
  button_calculate->setEnabled(true);
  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Extraction successful."));
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
