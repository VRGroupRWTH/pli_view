#include /* implements */ <ui/plugins/selector_plugin.hpp>

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
    on_change(offset(), size());
  });
  connect(slider_x          , &QxtSpanSlider::upperValueChanged, [&](int value)
  {
    line_edit_size_x  ->setText(QString::fromStdString(std::to_string(value - slider_x->lowerValue())));
    on_change(offset(), size());
  });
  connect(slider_y          , &QxtSpanSlider::lowerValueChanged, [&](int value)
  {
    line_edit_offset_y->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_y  ->setText(QString::fromStdString(std::to_string(slider_y->upperValue() - value)));
    on_change(offset(), size());
  });
  connect(slider_y          , &QxtSpanSlider::upperValueChanged, [&](int value)
  {
    line_edit_size_y  ->setText(QString::fromStdString(std::to_string(value - slider_y->lowerValue())));
    on_change(offset(), size());
  });
  connect(slider_z          , &QxtSpanSlider::lowerValueChanged, [&](int value)
  {
    line_edit_offset_z->setText(QString::fromStdString(std::to_string(value)));
    line_edit_size_z  ->setText(QString::fromStdString(std::to_string(slider_z->upperValue() - value)));
    on_change(offset(), size());
  });
  connect(slider_z          , &QxtSpanSlider::upperValueChanged, [&](int value)
  {
    line_edit_size_z  ->setText(QString::fromStdString(std::to_string(value - slider_z->lowerValue())));
    on_change(offset(), size());
  });
  connect(line_edit_offset_x, &QLineEdit::editingFinished      , [&]
  {
    auto value = line_edit_utility::get_text<std::size_t>(line_edit_offset_x);
    logger_ ->info         ("X offset is set to {}.", value);
    slider_x->setLowerValue(value);
    on_change(offset(), size());
  });
  connect(line_edit_offset_y, &QLineEdit::editingFinished      , [&]
  {
    auto value = line_edit_utility::get_text<std::size_t>(line_edit_offset_y);
    logger_ ->info         ("Y offset is set to {}.", value);
    slider_y->setLowerValue(value);
    on_change(offset(), size());
  });
  connect(line_edit_offset_z, &QLineEdit::editingFinished      , [&]
  {
    auto value = line_edit_utility::get_text<std::size_t>(line_edit_offset_z);
    logger_ ->info         ("Z offset is set to {}.", value);
    slider_z->setLowerValue(value);
    on_change(offset(), size());
  });
  connect(line_edit_size_x  , &QLineEdit::editingFinished      , [&]
  {
    auto value = line_edit_utility::get_text<std::size_t>(line_edit_size_x);
    logger_ ->info         ("X size is set to {}.", value);
    slider_x->setUpperValue(slider_x->lowerValue() + value);
    on_change(offset(), size());
  });
  connect(line_edit_size_y  , &QLineEdit::editingFinished      , [&]
  {
    auto value = line_edit_utility::get_text<std::size_t>(line_edit_size_y);
    logger_ ->info         ("Y size is set to {}.", value);
    slider_y->setUpperValue(slider_y->lowerValue() + value);
    on_change(offset(), size());
  });
  connect(line_edit_size_z  , &QLineEdit::editingFinished      , [&]
  {
    auto value = line_edit_utility::get_text<std::size_t>(line_edit_size_z);
    logger_ ->info         ("Z size is set to {}.", value);
    slider_z->setUpperValue(slider_z->lowerValue() + value);
    on_change(offset(), size());
  });
}

void selector_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));
  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    auto bounds = owner_->get_plugin<data_plugin>()->io()->load_fiber_direction_dataset_bounds();
    slider_x->setMinimum(bounds.first[0]); slider_x->setMaximum(bounds.second[0]);
    slider_y->setMinimum(bounds.first[1]); slider_y->setMaximum(bounds.second[1]);
    slider_z->setMinimum(bounds.first[2]); slider_z->setMaximum(bounds.second[2]);
    update();
  });

  logger_->info(std::string("Start successful."));
}

std::array<std::size_t, 3> selector_plugin::offset() const
{
  return
  {
    line_edit_utility::get_text<std::size_t>(line_edit_offset_x),
    line_edit_utility::get_text<std::size_t>(line_edit_offset_y),
    line_edit_utility::get_text<std::size_t>(line_edit_offset_z)
  };
}
std::array<std::size_t, 3> selector_plugin::size  () const
{
  return
  {
    line_edit_utility::get_text<std::size_t>(line_edit_size_x),
    line_edit_utility::get_text<std::size_t>(line_edit_size_y),
    line_edit_utility::get_text<std::size_t>(line_edit_size_z)
  };
}
}
