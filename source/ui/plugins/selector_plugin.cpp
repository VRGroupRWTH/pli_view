#include <pli_vis/ui/plugins/selector_plugin.hpp>

#include <algorithm>

#include <pli_vis/ui/window.hpp>
#include <pli_vis/utility/line_edit_utility.hpp>
#include <pli_vis/utility/qt_text_browser_sink.hpp>

namespace pli
{
selector_plugin::selector_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);

  connect(slider_x          , &QxtSpanSlider::lowerValueChanged, [&](int value)
  {
    line_edit_offset_x->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_x  ->setText(QString::fromStdString(std::to_string(slider_x->upperValue() - value)));
  });
  connect(slider_x          , &QxtSpanSlider::upperValueChanged, [&](int value)
  {
    line_edit_size_x->setText(QString::fromStdString(std::to_string(value - slider_x->lowerValue())));
  });
  connect(slider_x          , &QxtSpanSlider::sliderReleased, [&]
  {
    image->set_selection_offset_percentage({static_cast<float>(slider_x->lowerValue())                          / slider_x->maximum(), image->selection_offset_percentage()[1]});
    image->set_selection_size_percentage  ({static_cast<float>(slider_x->upperValue() - slider_x->lowerValue()) / slider_x->maximum(), image->selection_size_percentage  ()[1]});
  });
  connect(slider_y          , &QxtSpanSlider::lowerValueChanged, [&](int value)
  {
    line_edit_offset_y->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_y  ->setText(QString::fromStdString(std::to_string(slider_y->upperValue() - value)));
  });
  connect(slider_y          , &QxtSpanSlider::upperValueChanged, [&](int value)
  {
    line_edit_size_y  ->setText(QString::fromStdString(std::to_string(value - slider_y->lowerValue())));
  });
  connect(slider_y          , &QxtSpanSlider::sliderReleased, [&]
  {
    image->set_selection_offset_percentage({image->selection_offset_percentage()[0], static_cast<float>(slider_y->lowerValue())                          / slider_y->maximum()});
    image->set_selection_size_percentage  ({image->selection_size_percentage  ()[0], static_cast<float>(slider_y->upperValue() - slider_y->lowerValue()) / slider_y->maximum()});
  });
  connect(slider_z          , &QxtSpanSlider::lowerValueChanged, [&](int value)
  {
    line_edit_offset_z->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_z  ->setText(QString::fromStdString(std::to_string(slider_z->upperValue() - value)));
  });
  connect(slider_z          , &QxtSpanSlider::upperValueChanged, [&](int value)
  {
    line_edit_size_z  ->setText(QString::fromStdString(std::to_string(value - slider_z->lowerValue())));
  });
  connect(line_edit_offset_x, &QLineEdit::editingFinished      , [&]
  {
    auto value = std::max(std::min(line_edit_utility::get_text<int>(line_edit_offset_x), int(slider_x->maximum())), int(slider_x->minimum()));
    if (slider_x->upperValue() < value)
      slider_x->setUpperValue(value);
    slider_x->setLowerValue(value);
    image->set_selection_offset_percentage({static_cast<float>(value) / slider_x->maximum(), image->selection_offset_percentage()[1]});
    image->set_selection_size_percentage  ({static_cast<float>(slider_x->upperValue() - slider_x->lowerValue()) / slider_x->maximum(), image->selection_size_percentage  ()[1]});
    line_edit_size_x->setText(QString::fromStdString(std::to_string(slider_x->upperValue() - value)));
  });
  connect(line_edit_size_x  , &QLineEdit::editingFinished      , [&]
  {
    auto value = std::max(std::min(line_edit_utility::get_text<int>(line_edit_size_x), int(slider_x->maximum())), int(slider_x->minimum()));
    slider_x->setUpperValue(slider_x->lowerValue() + value);
    image->set_selection_size_percentage({static_cast<float>(value) / slider_x->maximum(), image->selection_size_percentage()[1]});
  });
  connect(line_edit_offset_y, &QLineEdit::editingFinished      , [&]
  {
    int value = std::max(std::min(line_edit_utility::get_text<int>(line_edit_offset_y), int(slider_y->maximum())), int(slider_y->minimum()));
    if (slider_y->upperValue() < value)
      slider_y->setUpperValue(value);
    slider_y->setLowerValue(value);
    image->set_selection_offset_percentage({image->selection_offset_percentage()[0], static_cast<float>(value) / slider_y->maximum()});
    image->set_selection_size_percentage  ({image->selection_size_percentage  ()[0], static_cast<float>(slider_y->upperValue() - slider_y->lowerValue()) / slider_y->maximum()});
    line_edit_size_y->setText(QString::fromStdString(std::to_string(slider_y->upperValue() - value)));
  });
  connect(line_edit_size_y  , &QLineEdit::editingFinished      , [&]
  {
    auto value = std::max(std::min(line_edit_utility::get_text<int>(line_edit_size_y), int(slider_y->maximum())), int(slider_y->minimum()));
    slider_y->setUpperValue(slider_y->lowerValue() + value);
    image->set_selection_size_percentage({image->selection_size_percentage()[0], static_cast<float>(value) / slider_y->maximum()});
  });
  connect(line_edit_offset_z, &QLineEdit::editingFinished      , [&]
  {
    auto value = std::max(std::min(line_edit_utility::get_text<int>(line_edit_offset_z), int(slider_z->maximum())), int(slider_z->minimum()));
    if (slider_z->upperValue() < value)
      slider_z->setUpperValue(value + 1);
    slider_z->setLowerValue(value);
    line_edit_size_z->setText(QString::fromStdString(std::to_string(slider_z->upperValue() - value)));
  });
  connect(line_edit_size_z  , &QLineEdit::editingFinished      , [&]
  {
    auto value = std::min(line_edit_utility::get_text<int>(line_edit_size_z), int(slider_z->maximum() - slider_z->minimum()));
    slider_z->setUpperValue(slider_z->lowerValue() + value);
  });
  connect(button_update     , &QAbstractButton::clicked        , [&]
  {
    on_change(selection_offset(), selection_size(), selection_stride());
  });

  connect(image, &overview_image::on_selection_change, [&](const std::array<float, 2> offset_perc, const std::array<float, 2> size_perc)
  {
    std::array<int, 2> offset { int(offset_perc[0] * slider_x->maximum()), int(offset_perc[1] * slider_y->maximum()) };
    std::array<int, 2> size   { int(size_perc  [0] * slider_x->maximum()), int(size_perc  [1] * slider_y->maximum()) };

    line_edit_offset_x->setText      (QString::fromStdString(std::to_string(offset[0])));
    line_edit_size_x  ->setText      (QString::fromStdString(std::to_string(size  [0])));
    slider_x          ->setLowerValue(offset[0]);
    slider_x          ->setUpperValue(offset[0] + size[0]);
    line_edit_offset_y->setText      (QString::fromStdString(std::to_string(offset[1])));
    line_edit_size_y  ->setText      (QString::fromStdString(std::to_string(size  [1])));
    slider_y          ->setLowerValue(offset[1]);
    slider_y          ->setUpperValue(offset[1] + size[1]);
  });
}

std::array<std::size_t, 3> selector_plugin::selection_offset() const
{
  return
  {
    line_edit_utility::get_text<std::size_t>(line_edit_offset_x),
    line_edit_utility::get_text<std::size_t>(line_edit_offset_y),
    line_edit_utility::get_text<std::size_t>(line_edit_offset_z)
  };
}
std::array<std::size_t, 3> selector_plugin::selection_size  () const
{
  return
  {
    line_edit_utility::get_text<std::size_t>(line_edit_size_x),
    line_edit_utility::get_text<std::size_t>(line_edit_size_y),
    line_edit_utility::get_text<std::size_t>(line_edit_size_z)
  };
}
std::array<std::size_t, 3> selector_plugin::selection_stride() const
{
  return
  {
    line_edit_utility::get_text<std::size_t>(line_edit_stride_x),
    line_edit_utility::get_text<std::size_t>(line_edit_stride_y),
    line_edit_utility::get_text<std::size_t>(line_edit_stride_z)
  };
}

void selector_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    logger_->info(std::string("Updating selector..."));

    auto io = owner_->get_plugin<data_plugin>()->io();
    if (io)
    {
      // Adjust slider boundaries.
      auto bounds = io->load_retardation_dataset_bounds();
      slider_x->setMinimum(int(bounds.first[0])); slider_x->setMaximum(int(bounds.second[0]));
      slider_y->setMinimum(int(bounds.first[1])); slider_y->setMaximum(int(bounds.second[1]));
      slider_z->setMinimum(int(bounds.first[2])); slider_z->setMaximum(int(bounds.second[2]));
      slider_z->setSpan   (int(bounds.first[2]) , int(bounds.first[2] + 1));

      // Generate preview image.
      std::array<std::size_t, 3> size   = {1, 1, 1};
      std::array<std::size_t, 3> stride = {1, 1, 1};
      size  [0] = std::min(int(bounds.second[0]), 2048);
      stride[0] = bounds.second[0] / size  [0];
      size  [1] = bounds.second[1] / stride[0];
      stride[1] = stride[0];
      auto data = io->load_retardation_dataset(bounds.first, size, stride);
      boost::multi_array<unsigned char, 3> converted(boost::extents[size[0]][size[1]][size[2]], boost::fortran_storage_order());
      for(auto i = 0; i < size[0]; i++)
        for(auto j = 0; j < size[1]; j++)
          for (auto k = 0; k < size[2]; k++)
            converted[i][j][k] = data[i][j][k] * 255.0;
      image->setPixmap(QPixmap::fromImage(QImage(converted.data(), int(size[0]), int(size[1]), QImage::Format::Format_Grayscale8)));

      // Adjust widget size.
      image    ->setSizeIncrement(int(size[0]), int(size[1]));
      letterbox->setWidget       (image);
      image    ->update          ();
    }
    else
    {
      slider_x->setMinimum(0); slider_x->setMaximum(0);
      slider_y->setMinimum(0); slider_y->setMaximum(0);
      slider_z->setMinimum(0); slider_z->setMaximum(0);

      image->setPixmap(QPixmap());
    }

    update();
    image->update();

    logger_->info(std::string("Update successful."));
  });

  logger_->info(std::string("Start successful."));
}
}
