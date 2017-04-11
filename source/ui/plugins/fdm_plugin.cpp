#include /* implements */ <ui/plugins/fdm_plugin.hpp>

#include <functional>
#include <future>
#include <limits>

#include <boost/optional.hpp>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>
#include <visualization/odf_field.hpp>

namespace pli
{
fdm_plugin::fdm_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
         
  line_edit_offset_x ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_y ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_z ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_x   ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_y   ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_z   ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  
  line_edit_samples_x->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_samples_y->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
                                
  connect(checkbox_enabled           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string(state ? "Enabled." : "Disabled."));
    odf_field_->set_active(state);
  });

  connect(line_edit_offset_x         , &QLineEdit::editingFinished, [&] 
  {
    logger_->info("X offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_x));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_offset_y         , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Y offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_y));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_offset_z         , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Z offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_z));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_size_x           , &QLineEdit::editingFinished, [&] 
  {
    logger_->info("X size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_x));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_size_y           , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Y size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_y));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_size_z           , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Z size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_z));
    if (checkbox_auto_update->isChecked())
      update();
  });
       
  connect(line_edit_samples_x        , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Longitude partitions are set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_samples_x));
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(line_edit_samples_y        , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Latitude partitions are set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_samples_y));
    if (checkbox_auto_update->isChecked())
      update();
  });
                                     
  connect(checkbox_depth_0           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Grid depth 0 is {}.", state ? "enabled" : "disabled");
    select_depths();
  });
  connect(checkbox_depth_1           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Grid depth 1 is {}.", state ? "enabled" : "disabled");
    select_depths();
  });
  connect(checkbox_depth_2           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Grid depth 2 is {}.", state ? "enabled" : "disabled");
    select_depths();
  });
  connect(checkbox_depth_3           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Grid depth 3 is {}.", state ? "enabled" : "disabled");
    select_depths();
  });
  connect(checkbox_depth_4           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Grid depth 4 is {}.", state ? "enabled" : "disabled");
    select_depths();
  });
  connect(checkbox_depth_5           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Grid depth 5 is {}.", state ? "enabled" : "disabled");
    select_depths();
  });
  connect(checkbox_depth_6           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Grid depth 6 is {}.", state ? "enabled" : "disabled");
    select_depths();
  });
  connect(checkbox_depth_7           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Grid depth 7 is {}.", state ? "enabled" : "disabled");
    select_depths();
  });
  connect(checkbox_depth_8           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Grid depth 8 is {}.", state ? "enabled" : "disabled");
    select_depths();
  });
  connect(checkbox_depth_9           , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Grid depth 9 is {}.", state ? "enabled" : "disabled");
    select_depths();
  });

  connect(checkbox_clustering_enabled, &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Clustering is {}.", state ? "enabled" : "disabled");
    if (checkbox_auto_update->isChecked())
      update();
  });
  connect(slider_clustering_threshold, &QSlider::sliderReleased   , [&]()
  {
    logger_->info("Clustering threshold is set to {}.", float(slider_clustering_threshold->value()) / 100.0);
    if (checkbox_auto_update->isChecked())
      update();
  });
  
  connect(checkbox_auto_update       , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info("Auto update is {}.", state ? "enabled" : "disabled");
    button_update->setEnabled(!state);
    if (state)
      update();
  });
  connect(button_update              , &QPushButton::clicked      , [&]
  {
    update();
  });

}

void fdm_plugin::start        ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    update();
  });

  odf_field_ = owner_->viewer->add_renderable<odf_field>();

  select_depths();

  logger_->info(std::string("Start successful."));
}
void fdm_plugin::update       () const
{
  logger_->info(std::string("Updating viewer..."));

  auto data_plugin = owner_->get_plugin<pli::data_plugin>();
  auto io          = data_plugin->io();
  if  (io == nullptr)
  {
    logger_->info(std::string("Update failed: No data."));
    return;
  }

  owner_->viewer->set_wait_spinner_enabled(true);
  
  std::array<std::size_t, 3> offset =
  {line_edit_utility::get_text<std::size_t>(line_edit_offset_x),
   line_edit_utility::get_text<std::size_t>(line_edit_offset_y),
   line_edit_utility::get_text<std::size_t>(line_edit_offset_z)};
  std::array<std::size_t, 3> size =
  {line_edit_utility::get_text<std::size_t>(line_edit_size_x),
   line_edit_utility::get_text<std::size_t>(line_edit_size_y),
   line_edit_utility::get_text<std::size_t>(line_edit_size_z)};
  uint2 tessellations =
  {line_edit_utility::get_text<std::size_t>(line_edit_samples_x),
   line_edit_utility::get_text<std::size_t>(line_edit_samples_y)};

  std::array<float      , 3>                    spacing      ;
  std::array<std::size_t, 3>                    block_size   ;
  boost::optional<boost::multi_array<float, 4>> distributions;

  std::future<void> result(std::async(std::launch::async, [&]()
  {
    try
    {
      if(checkbox_enabled)
      {
        spacing    = io->load_vector_spacing();
        block_size = io->load_block_size    ();
        distributions.reset(io->load_fiber_distribution_dataset(offset, size));

        // Roll dimensions to power of two.
        size[0] = pow(2, ceil(log(size[0]) / log(2)));
        size[1] = pow(2, ceil(log(size[1]) / log(2)));
        size[2] = pow(2, ceil(log(size[2]) / log(2)));
        (*distributions).resize(boost::extents[size[0]][size[1]][size[2]][(*distributions).shape()[3]]);
      }
    }
    catch (std::exception& exception)
    {
      logger_->error(std::string(exception.what()));
    }
  }));

  while (result.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    QApplication::processEvents();

  if (distributions.is_initialized() && distributions.get().num_elements() > 0)
  {
    auto shape = distributions.get().shape();
    uint3  cuda_size      {unsigned(shape[0]), unsigned(shape[1]), unsigned(shape[2])};
    float3 cuda_spacing   {spacing   [0], spacing   [1], spacing   [2]};
    uint3  cuda_block_size{block_size[0], block_size[1], block_size[2]};

    odf_field_->set_data(
      cuda_size, 
      shape[3], 
      distributions.get().data(), 
      tessellations, 
      cuda_spacing,
      cuda_block_size, 
      1.0, 
      checkbox_clustering_enabled->isChecked(),
      float(slider_clustering_threshold->value()) / 100.0F,
      [&](const std::string& message)
      {
        logger_->info(message);
      });
  }

  owner_->viewer->set_wait_spinner_enabled(false);
  owner_->viewer->update();

  logger_->info(std::string("Update successful."));
}

void fdm_plugin::select_depths() const
{
  logger_->info(std::string("Selecting depths."));

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

  logger_->info(std::string("Selection successful."));
}
}
