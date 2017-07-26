#include <pli_vis/ui/plugins/data_plugin.hpp>

#define _USE_MATH_DEFINES

#include <math.h>

#include <QFileDialog>

#include <pli_vis/cuda/sh/convert.h>
#include <pli_vis/cuda/sh/vector_ops.h>
#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
data_plugin::data_plugin(QWidget* parent) 
: plugin(parent)
, io_(line_edit::get_text(line_edit_file         ),
      line_edit::get_text(line_edit_transmittance),
      line_edit::get_text(line_edit_retardation  ),
      line_edit::get_text(line_edit_direction    ),
      line_edit::get_text(line_edit_inclination  ),
      line_edit::get_text(line_edit_mask         ),
      line_edit::get_text(line_edit_unit_vector  ))
{
  connect(button_browse          , &QPushButton::clicked      , [&]
  {
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("HDF5 Files (*.h5)")).toStdString();
    
    io_.set_filepath       (filename.c_str());
    line_edit_file->setText(filename.c_str());

    if(!filename.empty())
    {      
      transmittance_bounds_ = io_.load_transmittance_bounds();
      retardation_bounds_   = io_.load_retardation_bounds  ();
      direction_bounds_     = io_.load_direction_bounds    ();
      inclination_bounds_   = io_.load_inclination_bounds  ();
      mask_bounds_          = io_.load_mask_bounds         ();
      unit_vector_bounds_   = io_.load_unit_vector_bounds  ();
      logger_->info("Successfully opened file: {}.", filename);
    }
    else
    {
      logger_->info(std::string("Failed to open file: No filepath."));
    }

    on_change();
  });
  connect(line_edit_transmittance, &QLineEdit::editingFinished, [&] 
  {
    io_.set_transmittance_path(line_edit::get_text(line_edit_transmittance));
    on_change();
  });
  connect(line_edit_retardation  , &QLineEdit::editingFinished, [&]
  {
    io_.set_retardation_path(line_edit::get_text(line_edit_retardation));
    on_change();
  });
  connect(line_edit_direction    , &QLineEdit::editingFinished, [&]
  {
    io_.set_direction_path(line_edit::get_text(line_edit_direction));
    on_change();
  });
  connect(line_edit_inclination  , &QLineEdit::editingFinished, [&]
  {
    io_.set_inclination_path(line_edit::get_text(line_edit_inclination));
    on_change();
  });
  connect(line_edit_mask         , &QLineEdit::editingFinished, [&]
  {
    io_.set_mask_path(line_edit::get_text(line_edit_mask));
    on_change();
  });
  connect(line_edit_unit_vector  , &QLineEdit::editingFinished, [&]
  {
    io_.set_unit_vector_path(line_edit::get_text(line_edit_unit_vector));
    on_change();
  });
  connect(button_vervet_defaults , &QPushButton::clicked      , [&]
  {
    line_edit_file         ->setPlaceholderText("C:/Vervet1818.h5");
    line_edit_transmittance->setPlaceholderText("%Slice%/Microscope/Processed/Registered/NTransmittance");
    line_edit_retardation  ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Retardation");
    line_edit_direction    ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Direction");
    line_edit_inclination  ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Inclination");
    line_edit_mask         ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/Mask");
    line_edit_unit_vector  ->setPlaceholderText("%Slice%/Microscope/Processed/Registered/UnitVector");
    io_.set_transmittance_path(line_edit::get_text(line_edit_transmittance));
    io_.set_retardation_path  (line_edit::get_text(line_edit_retardation  ));
    io_.set_direction_path    (line_edit::get_text(line_edit_direction    ));
    io_.set_inclination_path  (line_edit::get_text(line_edit_inclination  ));
    io_.set_mask_path         (line_edit::get_text(line_edit_mask         ));
    io_.set_unit_vector_path  (line_edit::get_text(line_edit_unit_vector  ));
    on_change();
  });
  connect(button_msa_defaults    , &QPushButton::clicked      , [&]
  {
    line_edit_file         ->setPlaceholderText("C:/MSA0309_s0536-0695.h5");
    line_edit_transmittance->setPlaceholderText("Transmittance");
    line_edit_retardation  ->setPlaceholderText("Retardation");
    line_edit_direction    ->setPlaceholderText("Direction");
    line_edit_inclination  ->setPlaceholderText("Inclination");
    line_edit_mask         ->setPlaceholderText("Mask");
    line_edit_unit_vector  ->setPlaceholderText("UnitVector");
    io_.set_transmittance_path(line_edit::get_text(line_edit_transmittance));
    io_.set_retardation_path  (line_edit::get_text(line_edit_retardation  ));
    io_.set_direction_path    (line_edit::get_text(line_edit_direction    ));
    io_.set_inclination_path  (line_edit::get_text(line_edit_inclination  ));
    io_.set_mask_path         (line_edit::get_text(line_edit_mask         ));
    io_.set_unit_vector_path  (line_edit::get_text(line_edit_unit_vector  ));
    on_change();
  });
}

void data_plugin::start()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<selector_plugin>(), &selector_plugin::on_change, [&] (
    const std::array<std::size_t, 3>& offset,
    const std::array<std::size_t, 3>& size  ,
    const std::array<std::size_t, 3>& stride)
  {
    owner_->viewer ->set_wait_spinner_enabled(true);
    owner_->toolbox->setEnabled(false);

    future_ = std::async(std::launch::async, [&]
    {
      try
      {
        transmittance_bounds_ = io_.load_transmittance_bounds();
        retardation_bounds_   = io_.load_retardation_bounds  ();
        direction_bounds_     = io_.load_direction_bounds    ();
        inclination_bounds_   = io_.load_inclination_bounds  ();
        mask_bounds_          = io_.load_mask_bounds         ();
        unit_vector_bounds_   = io_.load_unit_vector_bounds  ();

        transmittance_ = std::make_unique<boost::multi_array<float, 3>>(io_.load_transmittance(offset, size, stride));
        retardation_   = std::make_unique<boost::multi_array<float, 3>>(io_.load_retardation  (offset, size, stride));
        direction_     = std::make_unique<boost::multi_array<float, 3>>(io_.load_direction    (offset, size, stride));
        inclination_   = std::make_unique<boost::multi_array<float, 3>>(io_.load_inclination  (offset, size, stride));
        mask_          = std::make_unique<boost::multi_array<float, 3>>(io_.load_mask         (offset, size, stride));
        unit_vector_   = std::make_unique<boost::multi_array<float, 4>>(io_.load_unit_vector  (offset, size, stride));
      }
      catch (std::exception& exception)
      {
        logger_->error(std::string(exception.what()));
      }
    });

    while(future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
      QApplication::processEvents();

    on_load();

    owner_->toolbox->setEnabled(true);
    owner_->viewer ->set_wait_spinner_enabled(false);
    owner_->viewer ->update();
  });
}

boost::multi_array<unsigned char, 2> data_plugin::generate_preview_image(std::size_t x_resolution)
{
  if (retardation_bounds_.second[0] == 0)
    return boost::multi_array<unsigned char, 2>();

  std::array<std::size_t, 3> size   = {1, 1, 1};
  std::array<std::size_t, 3> stride = {1, 1, 1};
  size  [0] = std::min(int(retardation_bounds_.second[0]), int(x_resolution));
  stride[0] = retardation_bounds_.second[0] / size  [0];
  size  [1] = retardation_bounds_.second[1] / stride[0];
  stride[1] = stride[0];

  boost::multi_array<unsigned char, 2> preview_image(boost::extents[size[0]][size[1]], boost::fortran_storage_order());
  auto data = io_.load_retardation(retardation_bounds_.first, size, stride);
  for (auto x = 0; x < size[0]; x++)
    for (auto y = 0; y < size[1]; y++)
      preview_image[x][y] = data[x][y][0] * 255.0;
  return preview_image;
}
boost::multi_array<float3, 3>        data_plugin::generate_vectors      (bool        cartesian   )
{
  // Generate from direction - inclination pairs.
  if(direction_bounds_.second[0] != 0 && inclination_bounds_.second[0] != 0)
  {
    boost::multi_array<float3, 3> vectors(boost::extents[direction_->shape()[0]][direction_->shape()[1]][direction_->shape()[2]]);
    
    std::transform(
      direction_  ->data(), 
      direction_  ->data() + direction_->num_elements(), 
      inclination_->data(), 
      vectors     . data(), 
      [cartesian] (const float& direction, const float& inclination)
      {
        float3 vector {1.0, (90.0F + direction) * M_PI / 180.0F, (90.0F - inclination) * M_PI / 180.0F};
        return cartesian ? to_cartesian_coords(vector) : vector;
      });

    return vectors;
  }
  // Generate from unit vectors.
  else
  {
    boost::multi_array<float3, 3> vectors(boost::extents[unit_vector_->shape()[0]][unit_vector_->shape()[1]][unit_vector_->shape()[2]]);

    std::transform(
      reinterpret_cast<float3*>(unit_vector_->data()), 
      reinterpret_cast<float3*>(unit_vector_->data() + unit_vector_->num_elements()), 
      vectors.data(), 
      [cartesian] (const float3& unit_vector)
      {
        auto vector = unit_vector / length(unit_vector);
        return cartesian ? vector : to_spherical_coords(vector);
      });

    return vectors;
  }
}
}
