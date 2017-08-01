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
data_plugin::data_plugin(QWidget* parent) : plugin(parent)
{
  connect(button_open_slice , &QPushButton::clicked             , [&]
  {
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("HDF5 Files (*.h5)")).toStdString();
    if(filename.empty())
      return;

    io_.set_filepath          (filename.c_str());
    io_.set_transmittance_path("%Slice%/Microscope/Processed/Registered/NTransmittance");
    io_.set_retardation_path  ("%Slice%/Microscope/Processed/Registered/Retardation"   );
    io_.set_direction_path    ("%Slice%/Microscope/Processed/Registered/Direction"     );
    io_.set_inclination_path  ("%Slice%/Microscope/Processed/Registered/Inclination"   );
    io_.set_mask_path         ("%Slice%/Microscope/Processed/Registered/Mask"          );
    io_.set_unit_vector_path  ("%Slice%/Microscope/Processed/Registered/UnitVector"    );

    setup();
  });
  connect(button_open_volume, &QPushButton::clicked             , [&]
  {
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("HDF5 Files (*.h5)")).toStdString();
    if (filename.empty())
      return;

    io_.set_filepath          (filename.c_str());
    io_.set_transmittance_path("Transmittance");
    io_.set_retardation_path  ("Retardation"  );
    io_.set_direction_path    ("Direction"    );
    io_.set_inclination_path  ("Inclination"  );
    io_.set_mask_path         ("Mask"         );
    io_.set_unit_vector_path  ("UnitVector"   );

    setup();
  });
  connect(image             , &roi_selector::on_selection_change, [&](const std::array<float, 2> offset_perc, const std::array<float, 2> size_perc)
  {
    std::array<int, 2> offset { int(offset_perc[0] * slider_x->maximum()), int(offset_perc[1] * slider_y->maximum()) };
    std::array<int, 2> size   { int(size_perc  [0] * slider_x->maximum()), int(size_perc  [1] * slider_y->maximum()) };
    line_edit_offset_x->setText      (QString::fromStdString(std::to_string(offset[0])));
    line_edit_size_x  ->setText      (QString::fromStdString(std::to_string(size  [0])));
    line_edit_offset_y->setText      (QString::fromStdString(std::to_string(offset[1])));
    line_edit_size_y  ->setText      (QString::fromStdString(std::to_string(size  [1])));
    slider_x          ->setLowerValue(offset[0]);
    slider_x          ->setUpperValue(offset[0] + size[0]);
    slider_y          ->setLowerValue(offset[1]);
    slider_y          ->setUpperValue(offset[1] + size[1]);
  });
  connect(slider_x          , &QxtSpanSlider::lowerValueChanged , [&](int value)
  {
    line_edit_offset_x->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_x  ->setText(QString::fromStdString(std::to_string(slider_x->upperValue() - value)));
  });
  connect(slider_x          , &QxtSpanSlider::upperValueChanged , [&](int value)
  {
    line_edit_size_x->setText(QString::fromStdString(std::to_string(value - slider_x->lowerValue())));
  });
  connect(slider_x          , &QxtSpanSlider::sliderReleased    , [&]
  {
    image->set_selection_offset_percentage({static_cast<float>(slider_x->lowerValue())                          / slider_x->maximum(), image->selection_offset_percentage()[1]});
    image->set_selection_size_percentage  ({static_cast<float>(slider_x->upperValue() - slider_x->lowerValue()) / slider_x->maximum(), image->selection_size_percentage  ()[1]});
  });
  connect(slider_y          , &QxtSpanSlider::lowerValueChanged , [&](int value)
  {
    line_edit_offset_y->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_y  ->setText(QString::fromStdString(std::to_string(slider_y->upperValue() - value)));
  });
  connect(slider_y          , &QxtSpanSlider::upperValueChanged , [&](int value)
  {
    line_edit_size_y->setText(QString::fromStdString(std::to_string(value - slider_y->lowerValue())));
  });
  connect(slider_y          , &QxtSpanSlider::sliderReleased    , [&]
  {
    image->set_selection_offset_percentage({image->selection_offset_percentage()[0], static_cast<float>(slider_y->lowerValue())                          / slider_y->maximum()});
    image->set_selection_size_percentage  ({image->selection_size_percentage  ()[0], static_cast<float>(slider_y->upperValue() - slider_y->lowerValue()) / slider_y->maximum()});
  });
  connect(slider_z          , &QxtSpanSlider::lowerValueChanged , [&](int value)
  {
    line_edit_offset_z->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_z  ->setText(QString::fromStdString(std::to_string(slider_z->upperValue() - value)));
  });
  connect(slider_z          , &QxtSpanSlider::upperValueChanged , [&](int value)
  {
    line_edit_size_z->setText(QString::fromStdString(std::to_string(value - slider_z->lowerValue())));
  });
  connect(line_edit_offset_x, &QLineEdit::editingFinished       , [&]
  {
    auto value = std::max(std::min(line_edit::get_text<int>(line_edit_offset_x), int(slider_x->maximum())), int(slider_x->minimum()));
    if (slider_x->upperValue() < value)
      slider_x->setUpperValue(value);
    slider_x        ->setLowerValue                  (value);
    image           ->set_selection_offset_percentage({static_cast<float>(value) / slider_x->maximum(), image->selection_offset_percentage()[1]});
    image           ->set_selection_size_percentage  ({static_cast<float>(slider_x->upperValue() - slider_x->lowerValue()) / slider_x->maximum(), image->selection_size_percentage()[1]});
    line_edit_size_x->setText                        (QString::fromStdString(std::to_string(slider_x->upperValue() - value)));
  });
  connect(line_edit_size_x  , &QLineEdit::editingFinished       , [&]
  {
    auto value = std::max(std::min(line_edit::get_text<int>(line_edit_size_x), int(slider_x->maximum())), int(slider_x->minimum()));
    slider_x->setUpperValue                (slider_x->lowerValue() + value);
    image   ->set_selection_size_percentage({static_cast<float>(value) / slider_x->maximum(), image->selection_size_percentage()[1]});
  });
  connect(line_edit_offset_y, &QLineEdit::editingFinished       , [&]
  {
    auto value = std::max(std::min(line_edit::get_text<int>(line_edit_offset_y), int(slider_y->maximum())), int(slider_y->minimum()));
    if (slider_y->upperValue() < value)
      slider_y->setUpperValue(value);
    slider_y        ->setLowerValue                  (value);
    image           ->set_selection_offset_percentage({image->selection_offset_percentage()[0], static_cast<float>(value) / slider_y->maximum()});
    image           ->set_selection_size_percentage  ({image->selection_size_percentage  ()[0], static_cast<float>(slider_y->upperValue() - slider_y->lowerValue()) / slider_y->maximum()});
    line_edit_size_y->setText                        (QString::fromStdString(std::to_string(slider_y->upperValue() - value)));
  });
  connect(line_edit_size_y  , &QLineEdit::editingFinished       , [&]
  {
    auto value = std::max(std::min(line_edit::get_text<int>(line_edit_size_y), int(slider_y->maximum())), int(slider_y->minimum()));
    slider_y->setUpperValue                (slider_y->lowerValue() + value);
    image   ->set_selection_size_percentage({image->selection_size_percentage()[0], static_cast<float>(value) / slider_y->maximum()});
  });
  connect(line_edit_offset_z, &QLineEdit::editingFinished       , [&]
  {
    auto value = std::max(std::min(line_edit::get_text<int>(line_edit_offset_z), int(slider_z->maximum())), int(slider_z->minimum()));
    if (slider_z->upperValue() < value)
      slider_z->setUpperValue(value + 1);
    slider_z        ->setLowerValue(value);
    line_edit_size_z->setText      (QString::fromStdString(std::to_string(slider_z->upperValue() - value)));
  });
  connect(line_edit_size_z  , &QLineEdit::editingFinished       , [&]
  {
    auto value = std::min(line_edit::get_text<int>(line_edit_size_z), int(slider_z->maximum() - slider_z->minimum()));
    slider_z->setUpperValue(slider_z->lowerValue() + value);
  });
  connect(button_update     , &QAbstractButton::clicked         , [&]
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

        transmittance_        = std::make_unique<boost::multi_array<float, 3>>(io_.load_transmittance(selection_offset(), selection_size(), selection_stride()));
        retardation_          = std::make_unique<boost::multi_array<float, 3>>(io_.load_retardation  (selection_offset(), selection_size(), selection_stride()));
        direction_            = std::make_unique<boost::multi_array<float, 3>>(io_.load_direction    (selection_offset(), selection_size(), selection_stride()));
        inclination_          = std::make_unique<boost::multi_array<float, 3>>(io_.load_inclination  (selection_offset(), selection_size(), selection_stride()));
        mask_                 = std::make_unique<boost::multi_array<float, 3>>(io_.load_mask         (selection_offset(), selection_size(), selection_stride()));
        unit_vector_          = std::make_unique<boost::multi_array<float, 4>>(io_.load_unit_vector  (selection_offset(), selection_size(), selection_stride()));
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

std::array<std::size_t, 3> data_plugin::selection_offset() const
{
  return
  {
    line_edit::get_text<std::size_t>(line_edit_offset_x),
    line_edit::get_text<std::size_t>(line_edit_offset_y),
    line_edit::get_text<std::size_t>(line_edit_offset_z)
  };
}
std::array<std::size_t, 3> data_plugin::selection_size  () const
{
  auto stride = selection_stride();
  return
  {
    line_edit::get_text<std::size_t>(line_edit_size_x) / stride[0],
    line_edit::get_text<std::size_t>(line_edit_size_y) / stride[1],
    line_edit::get_text<std::size_t>(line_edit_size_z) / stride[2]
  };
}
std::array<std::size_t, 3> data_plugin::selection_stride() const
{
  return
  {
    line_edit::get_text<std::size_t>(line_edit_stride_x),
    line_edit::get_text<std::size_t>(line_edit_stride_y),
    line_edit::get_text<std::size_t>(line_edit_stride_z)
  };
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

    if(mask_bounds_.second[0] != 0)
      std::transform(
        vectors. data(),
        vectors. data() + vectors.num_elements(),
        mask_  ->data(),
        vectors. data(), 
        [] (const float3& vector, const float& mask)
        {
          return mask ? vector : float3{0.0F, 0.0F, 0.0F};
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
      
    if(mask_bounds_.second[0] != 0)
      std::transform(
        vectors. data(),
        vectors. data() + vectors.num_elements(),
        mask_  ->data(),
        vectors. data(), 
        [] (const float3& vector, const float& mask)
        {
          return mask ? vector : float3{0.0F, 0.0F, 0.0F};
        });

    return vectors;
  }
}

void data_plugin::start()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));
}
void data_plugin::setup()
{
  // Load bounds.
  transmittance_bounds_ = io_.load_transmittance_bounds();
  retardation_bounds_   = io_.load_retardation_bounds  ();
  direction_bounds_     = io_.load_direction_bounds    ();
  inclination_bounds_   = io_.load_inclination_bounds  ();
  mask_bounds_          = io_.load_mask_bounds         ();
  unit_vector_bounds_   = io_.load_unit_vector_bounds  ();

  // Adjust slider boundaries.
  auto bounds = retardation_bounds();
  slider_x->setMinimum(bounds.first[0]); slider_x->setMaximum(bounds.second[0]);
  slider_y->setMinimum(bounds.first[1]); slider_y->setMaximum(bounds.second[1]);
  slider_z->setMinimum(bounds.first[2]); slider_z->setMaximum(bounds.second[2]);
  slider_z->setSpan   (bounds.first[2], bounds.first[2] + 1);

  // Generate preview image.
  auto preview_image = generate_preview_image();
  auto shape         = preview_image.shape();
  image->setPixmap(QPixmap::fromImage(QImage(preview_image.data(), shape[0], shape[1], QImage::Format::Format_Grayscale8)));

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
}
}
