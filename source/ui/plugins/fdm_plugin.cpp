#include /* implements */ <ui/plugins/fdm_plugin.hpp>

#include <limits>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>
#include <visualization/odf_field.hpp>

namespace pli
{
fdm_plugin::fdm_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
         
  line_edit_offset_x        ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_y        ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_z        ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_x          ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_y          ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_z          ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  
  line_edit_samples_x       ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_samples_y       ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));

  line_edit_fom_offset_x    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_fom_offset_y    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_fom_offset_z    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_fom_size_x      ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_fom_size_y      ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_fom_size_z      ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_block_size_x    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_block_size_y    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_block_size_z    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_histogram_bins_x->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_histogram_bins_y->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_max_order       ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  
  connect(line_edit_offset_x        , &QLineEdit::editingFinished, [&] 
  {
    logger_->info("Selection X offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_x));
    update();
  });
  connect(line_edit_offset_y        , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Y offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_y));
    update();
  });
  connect(line_edit_offset_z        , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Z offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_z));
    update();
  });
  connect(line_edit_size_x          , &QLineEdit::editingFinished, [&] 
  {
    logger_->info("Selection X size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_x));
    update();
  });
  connect(line_edit_size_y          , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Y size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_y));
    update();
  });
  connect(line_edit_size_z          , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Z size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_z));
    update();
  });

  connect(checkbox_show             , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Show set to ") + (state ? "true" : "false"));
    update();
  });
  connect(line_edit_samples_x       , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Samples longitude partitions are set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_samples_x));
    update();
  });
  connect(line_edit_samples_y       , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Samples latitude partitions are set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_samples_y));
    update();
  });

  connect(checkbox_depth_0          , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Multiresolution grid depth 0 set to ") + (state ? "true" : "false"));
    select_depths();
  });
  connect(checkbox_depth_1          , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Multiresolution grid depth 1 set to ") + (state ? "true" : "false"));
    select_depths();
  });
  connect(checkbox_depth_2          , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Multiresolution grid depth 2 set to ") + (state ? "true" : "false"));
    select_depths();
  });
  connect(checkbox_depth_3          , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Multiresolution grid depth 3 set to ") + (state ? "true" : "false"));
    select_depths();
  });
  connect(checkbox_depth_4          , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Multiresolution grid depth 4 set to ") + (state ? "true" : "false"));
    select_depths();
  });
  connect(checkbox_depth_5          , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Multiresolution grid depth 5 set to ") + (state ? "true" : "false"));
    select_depths();
  });
  connect(checkbox_depth_6          , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Multiresolution grid depth 6 set to ") + (state ? "true" : "false"));
    select_depths();
  });
  connect(checkbox_depth_7          , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Multiresolution grid depth 7 set to ") + (state ? "true" : "false"));
    select_depths();
  });
  connect(checkbox_depth_8          , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Multiresolution grid depth 8 set to ") + (state ? "true" : "false"));
    select_depths();
  });
  connect(checkbox_depth_9          , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Multiresolution grid depth 9 set to ") + (state ? "true" : "false"));
    select_depths();
  });

  connect(line_edit_fom_offset_x    , &QLineEdit::editingFinished, [&] 
  {
    logger_->info("FOM X offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_fom_offset_x));
  });
  connect(line_edit_fom_offset_y    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("FOM Y offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_fom_offset_y));
  });
  connect(line_edit_fom_offset_z    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("FOM Z offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_fom_offset_z));
  });
  connect(line_edit_fom_size_x      , &QLineEdit::editingFinished, [&] 
  {
    logger_->info("FOM X size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_fom_size_x));
  });
  connect(line_edit_fom_size_y      , &QLineEdit::editingFinished, [&]
  {
    logger_->info("FOM Y size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_fom_size_y));
  });
  connect(line_edit_fom_size_z      , &QLineEdit::editingFinished, [&]
  {
    logger_->info("FOM Z size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_fom_size_z));
  });
  connect(line_edit_block_size_x    , &QLineEdit::editingFinished, [&] 
  {
    logger_->info("Block size X is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_block_size_x));
  });
  connect(line_edit_block_size_y    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Block size Y is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_block_size_y));
  });
  connect(line_edit_block_size_z    , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Block size Z is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_block_size_z));
  });
  connect(line_edit_histogram_bins_x, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Histogram latitude bins is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_histogram_bins_x));
  });
  connect(line_edit_histogram_bins_y, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Histogram longitude bins are set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_histogram_bins_y));
  });
  connect(line_edit_max_order       , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Maximum spherical harmonics order are set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_max_order));
  });
  connect(button_calculate          , &QPushButton::clicked      , [&]
  {
    calculate();
  });
  
  connect(checkbox_clustering_enabled, &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Clustering set to ") + (state ? "true" : "false"));
    update();
  });
  connect(slider_clustering_threshold, &QSlider::sliderReleased, [&]()
  {
    logger_->info("Clustering set to {}", float(slider_clustering_threshold->value()) / 100.0);
    update();
  });
}

void fdm_plugin::start    ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    logger_->info(std::string("Updating viewer."));
    update();
  });

  odf_field_ = owner_->viewer->add_renderable<odf_field>();
  select_depths();
}
void fdm_plugin::update   () const
{
  try
  {
    auto data_plugin = owner_->get_plugin<pli::data_plugin>();
    auto io          = data_plugin->io();

    if (io && checkbox_show->isChecked())
    {
      std::array<std::size_t, 3> offset = 
      { line_edit_utility::get_text<std::size_t>(line_edit_offset_x), 
        line_edit_utility::get_text<std::size_t>(line_edit_offset_y),
        line_edit_utility::get_text<std::size_t>(line_edit_offset_z)};
      
      std::array<std::size_t, 3> size = 
      { line_edit_utility::get_text<std::size_t>(line_edit_size_x),
        line_edit_utility::get_text<std::size_t>(line_edit_size_y),
        line_edit_utility::get_text<std::size_t>(line_edit_size_z)};
      
      uint2 tessellations =
      { line_edit_utility::get_text<std::size_t>(line_edit_samples_x), 
        line_edit_utility::get_text<std::size_t>(line_edit_samples_y)};

      auto fdm        = io->load_fiber_distribution_map(offset, size);

      // Roll dimensions to power of two.
      size[0] = pow(2, ceil(log(size[0]) / log(2)));
      size[1] = pow(2, ceil(log(size[1]) / log(2)));
      size[2] = pow(2, ceil(log(size[2]) / log(2)));
      fdm.resize(boost::extents[size[0]][size[1]][size[2]][fdm.shape()[3]]);

      auto shape      = fdm.shape              ();
      auto spacing    = io->load_vector_spacing();
      auto block_size = io->load_block_size    ();

      odf_field_->set_data(
        {shape[0], shape[1], shape[2]},
         shape[3],
         fdm.data(),
         tessellations,
        {spacing   [0], spacing   [1], spacing   [2]},
        {block_size[0], block_size[1], block_size[2]},
        1.0,
        checkbox_clustering_enabled->isChecked(),
        float(slider_clustering_threshold->value()) / 100.0);
    }
    
    owner_ ->viewer->update();
  }
  catch (std::exception& exception)
  {
    logger_->error(std::string(exception.what()));
  }
}

void fdm_plugin::select_depths() const
{
  odf_field_->set_visible_depths({
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

void fdm_plugin::calculate() const
{

}
}
