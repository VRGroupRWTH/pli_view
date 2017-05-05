#include /* implements */ <ui/plugins/selector_plugin.hpp>

#include <algorithm>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>

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
    line_edit_size_x  ->setText(QString::fromStdString(std::to_string(value - slider_x->lowerValue())));
  });
  connect(slider_x          , &QxtSpanSlider::sliderReleased   , [&]
  {
    on_change(selection_offset(), selection_size());
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
  connect(slider_y          , &QxtSpanSlider::sliderReleased   , [&]
  {
    on_change(selection_offset(), selection_size());
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
  connect(slider_z          , &QxtSpanSlider::sliderReleased   , [&]
  {
    on_change(selection_offset(), selection_size());
  });
  connect(line_edit_offset_x, &QLineEdit::editingFinished      , [&]
  {
    auto value = std::max(std::min(line_edit_utility::get_text<std::size_t>(line_edit_offset_x), std::size_t(slider_x->maximum())), std::size_t(slider_x->minimum()));
    logger_ ->info         ("X offset is set to {}.", value);
    slider_x->setLowerValue(value);
    on_change              (selection_offset(), selection_size());
  });
  connect(line_edit_size_x  , &QLineEdit::editingFinished      , [&]
  {
    auto value = std::max(std::min(line_edit_utility::get_text<std::size_t>(line_edit_size_x), std::size_t(slider_x->maximum())), std::size_t(slider_x->minimum()));
    logger_ ->info         ("X size is set to {}.", value);
    slider_x->setUpperValue(slider_x->lowerValue() + value);
    on_change              (selection_offset(), selection_size());
  });
  connect(line_edit_offset_y, &QLineEdit::editingFinished      , [&]
  {
    auto value = std::max(std::min(line_edit_utility::get_text<std::size_t>(line_edit_offset_y), std::size_t(slider_y->maximum())), std::size_t(slider_y->minimum()));
    logger_ ->info         ("Y offset is set to {}.", value);
    slider_y->setLowerValue(value);
    on_change              (selection_offset(), selection_size());
  });
  connect(line_edit_size_y  , &QLineEdit::editingFinished      , [&]
  {
    auto value = std::max(std::min(line_edit_utility::get_text<std::size_t>(line_edit_size_y), std::size_t(slider_y->maximum())), std::size_t(slider_y->minimum()));
    logger_ ->info         ("Y size is set to {}.", value);
    slider_y->setUpperValue(slider_y->lowerValue() + value);
    on_change              (selection_offset(), selection_size());
  });
  connect(line_edit_offset_z, &QLineEdit::editingFinished      , [&]
  {
    auto value = std::max(std::min(line_edit_utility::get_text<std::size_t>(line_edit_offset_z), std::size_t(slider_z->maximum())), std::size_t(slider_z->minimum()));
    logger_ ->info         ("Z offset is set to {}.", value);
    slider_z->setLowerValue(value);
    on_change              (selection_offset(), selection_size());
  });
  connect(line_edit_size_z  , &QLineEdit::editingFinished      , [&]
  {
    auto value = std::max(std::min(line_edit_utility::get_text<std::size_t>(line_edit_size_z), std::size_t(slider_z->maximum())), std::size_t(slider_z->minimum()));
    logger_ ->info         ("Z size is set to {}.", value);
    slider_z->setUpperValue(slider_z->lowerValue() + value);
    on_change              (selection_offset(), selection_size());
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

void selector_plugin::start ()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    upload();
  });

  logger_->info(std::string("Start successful."));
}

void selector_plugin::upload()
{
  logger_->info(std::string("Updating selector..."));

  auto io = owner_->get_plugin<data_plugin>()->io();
  if (io)
  {
    // Adjust slider boundaries.
    auto bounds = io->load_retardation_dataset_bounds();
    slider_x->setMinimum(bounds.first[0]); slider_x->setMaximum(bounds.second[0]);
    slider_y->setMinimum(bounds.first[1]); slider_y->setMaximum(bounds.second[1]);
    slider_z->setMinimum(bounds.first[2]); slider_z->setMaximum(bounds.second[2]);

    // Generate preview image.
    std::array<std::size_t, 3> size   = {1, 1, 1};
    std::array<std::size_t, 3> stride = {1, 1, 1};
    size  [0] = std::min(bounds.second[0], 256ull);
    stride[0] = bounds.second[0] / size  [0];
    size  [1] = bounds.second[1] / stride[0];
    stride[1] = stride[0];
    auto data = io->load_retardation_dataset(bounds.first, size, stride);
    boost::multi_array<unsigned char, 3> converted(boost::extents[size[0]][size[1]][size[2]], boost::fortran_storage_order());
    for(auto i = 0; i < size[0]; i++)
      for(auto j = 0; j < size[1]; j++)
        for (auto k = 0; k < size[2]; k++)
          converted[i][j][k] = data[i][j][k] * 255.0;
    image->setPixmap(QPixmap::fromImage(QImage(converted.data(), size[0], size[1], QImage::Format::Format_Grayscale8)));

    // Adjust widget size.
    image->setSizeIncrement(size[0], size[1]);
    letterbox->setWidget(image);
  }
  else
  {
    slider_x->setMinimum(0); slider_x->setMaximum(0);
    slider_y->setMinimum(0); slider_y->setMaximum(0);
    slider_z->setMinimum(0); slider_z->setMaximum(0);

    image->setPixmap(QPixmap());
  }

  update();

  logger_->info(std::string("Update successful."));
}
}
