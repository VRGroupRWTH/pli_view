#include /* implements */ <ui/plugins/fdm_plugin.hpp>

#define _USE_MATH_DEFINES

#include <math.h>
#include <limits>

#include <vtkProperty.h>

#include <convert.hpp>

#include <ui/window.hpp>
#include <utility/qt/line_edit_utility.hpp>
#include <utility/spdlog/qt_text_browser_sink.hpp>
#include <utility/vtk/fdm_factory.hpp>

template<typename precision = double>
precision factorial       (const unsigned int n)
{
  precision out(1.0);
  for (auto i = 2; i <= n; i++)
    out *= i;
  return out;
}
template<typename precision = double>
precision double_factorial(const unsigned int n)
{
  precision out(1.0);
  precision nd (n  );
  while (nd > precision(1.0))
  {
    out *= nd;
    nd  -= precision(2.0);
  }
  return out;
}

template<typename precision>
precision legendre(const int order, const int degree, const precision& x)
{
  precision pmm(1.0);
  if (degree > 0)
    pmm = (degree % 2 == 0 ? 1 : -1) * double_factorial<precision>(2 * degree - 1) * std::pow(1 - x * x, degree / 2.0);
  if (order == degree)
    return pmm;
  precision pmm1 = x * (2 * degree + 1) * pmm;
  if (order == degree + 1)
    return pmm1;
  for (auto n = degree + 2; n <= order; n++)
  {
    precision pmn = (x * (2 * n - 1) * pmm1 - (n + degree - 1) * pmm) / (n - degree);
    pmm = pmm1;
    pmm1 = pmn;
  }
  return pmm1;
}

template<typename input_precision, typename output_precision = input_precision>
output_precision evaluate_sh(
  const std::size_t&                    index ,
  const std::array<input_precision, 2>& angles)
{
  int l = std::floor(std::sqrt(index));
  int m = index - std::pow(l, 2) - l;

  output_precision kml = std::sqrt((2.0 * l + 1) * factorial<output_precision>(l - std::abs(m)) / (4.0 * M_PI * factorial<output_precision>(l + std::abs(m))));
  if (m > 0)
    return kml * std::sqrt(2.0) * std::cos( m * angles[0]) * legendre(l,  m, std::cos(angles[1]));
  if (m < 0)
    return kml * std::sqrt(2.0) * std::sin(-m * angles[0]) * legendre(l, -m, std::cos(angles[1]));
  return kml * legendre(l, 0, std::cos(angles[1]));
}

template<typename input_precision, typename output_precision = input_precision>
output_precision evaluate_sh_sum(
  const boost::detail::multi_array::const_sub_array<input_precision, 1>& coefficients,
  const std::array<input_precision, 2>&                                  angles      )
{
  output_precision sum = 0.0;
  for (auto i = 0; i < coefficients.size(); i++)
    sum += evaluate_sh(i, angles) * coefficients[i];
  return sum;
}

template<typename coefficient_type, typename sample_type = std::array<coefficient_type, 3>>
boost::multi_array<sample_type, 4> sample_coefficients(
  const boost::multi_array<coefficient_type, 4>& coefficients     ,
  const std::array<std::size_t, 2>&              sample_dimensions)
{
  auto shape = coefficients.shape();

  boost::multi_array<sample_type, 4> samples(boost::extents
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
        for (auto lon = 0; lon < sample_dimensions[0]; lon++)
        {
          for (auto lat = 0; lat < sample_dimensions[1]; lat++)
          {
            auto& sample = samples[x][y][z][lon * sample_dimensions[1] + lat];
            sample[1] = 2 * M_PI * coefficient_type(lon) /  sample_dimensions[0];
            sample[2] =     M_PI * coefficient_type(lat) / (sample_dimensions[1] - 1);
            sample[0] = evaluate_sh_sum(coefficients[x][y][z], std::array<coefficient_type, 2>{sample[1], sample[2]});
            sample    = pli::to_cartesian_coords(sample);
          }
        }
      }
    }
  }
  return samples;
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

      poly_data_ = fdm_factory::create(sample_coefficients(io->load_fiber_distribution_map(offset, size), sample_dimensions), sample_dimensions);
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
