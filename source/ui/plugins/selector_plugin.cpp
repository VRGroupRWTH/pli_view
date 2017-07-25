#include <pli_vis/ui/plugins/selector_plugin.hpp>

#include <algorithm>

#include <pli_vis/ui/utility/line_edit.hpp>
#include <pli_vis/ui/utility/text_browser_sink.hpp>
#include <pli_vis/ui/application.hpp>

namespace pli
{
selector_plugin::selector_plugin(QWidget* parent) : plugin(parent)
{
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
    on_change(selection_offset(), selection_size(), selection_stride());
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
}

std::array<std::size_t, 3> selector_plugin::selection_offset() const
{
  return
  {
    line_edit::get_text<std::size_t>(line_edit_offset_x),
    line_edit::get_text<std::size_t>(line_edit_offset_y),
    line_edit::get_text<std::size_t>(line_edit_offset_z)
  };
}
std::array<std::size_t, 3> selector_plugin::selection_size  () const
{
  auto stride = selection_stride();
  return
  {
    line_edit::get_text<std::size_t>(line_edit_size_x) / stride[0],
    line_edit::get_text<std::size_t>(line_edit_size_y) / stride[1],
    line_edit::get_text<std::size_t>(line_edit_size_z) / stride[2]
  };
}
std::array<std::size_t, 3> selector_plugin::selection_stride() const
{
  return
  {
    line_edit::get_text<std::size_t>(line_edit_stride_x),
    line_edit::get_text<std::size_t>(line_edit_stride_y),
    line_edit::get_text<std::size_t>(line_edit_stride_z)
  };
}

void selector_plugin::start ()
{
  set_sink(std::make_shared<text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    auto data_plugin = owner_->get_plugin<pli::data_plugin>();

    logger_->info(std::string("Updating selector..."));
    
    // Adjust slider boundaries.
    auto bounds = data_plugin->retardation_bounds();
    slider_x->setMinimum(bounds.first[0]); slider_x->setMaximum(bounds.second[0]);
    slider_y->setMinimum(bounds.first[1]); slider_y->setMaximum(bounds.second[1]);
    slider_z->setMinimum(bounds.first[2]); slider_z->setMaximum(bounds.second[2]);
    slider_z->setSpan   (bounds.first[2] , bounds.first[2] + 1);

    // Generate preview image.
    auto preview_image = data_plugin->generate_preview_image();
    auto shape         = preview_image.shape();
    image->setPixmap(QPixmap::fromImage(QImage(preview_image.data(), shape[0], shape[1], QImage::Format::Format_Grayscale8)));

    // Adjust widget size.
    image    ->setSizeIncrement(shape[0], shape[1]);
    letterbox->setWidget       (image);
    image    ->update          ();
    update();

    logger_->info(std::string("Update successful."));
  });
}
}
