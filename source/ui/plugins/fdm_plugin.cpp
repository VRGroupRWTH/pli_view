#include /* implements */ <ui/plugins/fdm_plugin.hpp>

#define _USE_MATH_DEFINES

#include <math.h>
#include <limits>

#include <vtkProperty.h>

#include <cush.h>
#include <convert.hpp>

#include <graphics/fdm_factory.hpp>
#include <graphics/sampling.hpp>
#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>

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
    update_viewer();
  });
  connect(line_edit_offset_y        , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Y offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_y));
    update_viewer();
  });
  connect(line_edit_offset_z        , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Z offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_z));
    update_viewer();
  });
  connect(line_edit_size_x          , &QLineEdit::editingFinished, [&] 
  {
    logger_->info("Selection X size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_x));
    update_viewer();
  });
  connect(line_edit_size_y          , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Y size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_y));
    update_viewer();
  });
  connect(line_edit_size_z          , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Z size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_z));
    update_viewer();
  });

  connect(checkbox_show             , &QCheckBox::stateChanged   , [&](int state)
  {
    logger_->info(std::string("Show set to ") + (state ? "true" : "false"));
    update_viewer();
  });
  connect(line_edit_samples_x       , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Samples longitude partitions are set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_samples_x));
    update_viewer();
  });
  connect(line_edit_samples_y       , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Samples latitude partitions are set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_samples_y));
    update_viewer();
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

  poly_data_ = vtkSmartPointer<vtkPolyData>      ::New();
  mapper_    = vtkSmartPointer<vtkPolyDataMapper>::New();
  actor_     = vtkSmartPointer<vtkActor>         ::New();
  actor_->SetMapper(mapper_);
  actor_->GetProperty()->SetLighting(false);
}

void fdm_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    logger_->info(std::string("Updating viewer."));
    update_viewer();
  });

  owner_->viewer->renderer()->AddActor(actor_);
  owner_->viewer->renderer()->ResetCamera();
}

void fdm_plugin::update_viewer()
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
      
      std::array<std::size_t, 2> sample_dimensions =
      { line_edit_utility::get_text<std::size_t>(line_edit_samples_x), 
        line_edit_utility::get_text<std::size_t>(line_edit_samples_y)};

      auto fdm     = io->load_fiber_distribution_map(offset, size);
      auto samples = sample_sums(fdm, sample_dimensions);
      poly_data_   = fdm_factory::create(samples, sample_dimensions);
    }
    else
      poly_data_ = vtkSmartPointer<vtkPolyData>::New();
    
    mapper_->SetInputData  (poly_data_);
    mapper_->Update        ();
    owner_ ->viewer->update();
  }
  catch (std::exception& exception)
  {
    logger_->error(std::string(exception.what()));
  }
}
void fdm_plugin::calculate    () const
{

}
}
