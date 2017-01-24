#include /* implements */ <ui/plugins/fdm_plugin.hpp>

#define _USE_MATH_DEFINES

#include <math.h>
#include <limits>

#include <ui/window.hpp>
#include <utility/qt/line_edit_utility.hpp>
#include <utility/spdlog/qt_text_browser_sink.hpp>
#include <utility/vtk/fdm_factory.hpp>

template<typename input_precision, typename output_precision>
output_precision evaluate_sh(
  const std::size_t&                    coefficient_index,
  const std::array<input_precision, 2>& angles           ,
  const std::array<std::size_t    , 2>& sample_dimensions,
  const input_precision&                weight           )
{
  return 0;
}

template<typename input_precision, typename output_precision = std::array<input_precision, 3>>
boost::multi_array<output_precision, 4> sample_coefficients(
  const boost::multi_array<input_precision, 4>& coefficients     ,
  const std  ::      array<std::size_t    , 2>& sample_dimensions)
{
  auto shape = coefficients.shape();

  boost::multi_array<output_precision, 4> samples(boost::extents
    [shape[0]]
    [shape[1]]
    [shape[2]]
    [sample_dimensions[0] * sample_dimensions[1]]);

  for (auto x = 0; x < shape[0]; x++)
  {
    for (auto y = 0; y < shape[1]; y++)
    {
      for (auto z = 0; z < shape[2]; z++)
      {
        for (auto c = 0; c < shape[3]; c++)
        {
          for (auto lon = 0; lon < sample_dimensions[0]; lon++)
          {
            for (auto lat = 0; lat < sample_dimensions[1]; lat++)
            {
              auto& sample = samples[x][y][z][lon * sample_dimensions[0] + lat];
              sample[0]    = 1;
              sample[1]    = 2 * M_PI / sample_dimensions[0] * (lon / sample_dimensions[0]);
              sample[2]    =     M_PI / sample_dimensions[1] * (lat / sample_dimensions[0]);
            }
          }
        }
      }
    }
  }

  return samples;

  //if (coefficient_index == 0)
  //{
  //  output_points[sample_index].y = 2 * M_PI / output_resolution * (sample_index / output_resolution);
  //  output_points[sample_index].z =     M_PI / output_resolution * (sample_index % output_resolution);
  //}
  //
  //atomicAdd(&output_points[sample_index].x, evaluate<input_precision>(
  //  coefficient_lm.x,
  //  coefficient_lm.y,
  //  2 * M_PI / output_resolution * (sample_index / output_resolution),
  //      M_PI / output_resolution * (sample_index % output_resolution),
  //  coefficients[coefficient_index]));
}

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
  });
  connect(line_edit_samples_y       , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Samples latitude partitions are set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_samples_y));
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
        line_edit_utility::get_text<std::size_t>(line_edit_samples_x)};

      poly_data_ = fdm_factory::create(sample_coefficients(io->load_fiber_distribution_map(offset, size), sample_dimensions), sample_dimensions);
    }
    else
      poly_data_ = vtkSmartPointer<vtkPolyData>::New();
    
    mapper_->SetInputData(poly_data_);
    actor_ ->SetMapper   (mapper_);
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
