#include <pli_vis/ui/plugins/color_plugin.hpp>

#include <boost/format.hpp>

#include <pli_vis/ui/utility/line_edit.hpp>

namespace pli
{
color_plugin::color_plugin(QWidget* parent) : plugin(parent)
{
  line_edit_k->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));

  connect(slider_k   , &QSlider::valueChanged     , [&]
  {
    line_edit_k->setText(QString::fromStdString((boost::format("%.4f") % (float(slider_k->value()) / slider_k->maximum())).str()));
  });
  connect(line_edit_k, &QLineEdit::editingFinished, [&]
  {
    auto value = line_edit::get_text<float>(line_edit_k);
    slider_k->setValue (value * slider_k->maximum());
  });
}

int   color_plugin::mode    () const
{
  if(radio_button_hsl_1->isChecked()) return 0;
  if(radio_button_hsl_2->isChecked()) return 1;
  if(radio_button_hsv_1->isChecked()) return 2;
  if(radio_button_hsv_2->isChecked()) return 3;
  if(radio_button_rgb  ->isChecked()) return 4;
  return -1;
}
float color_plugin::k       () const
{
  return line_edit::get_text<float>(line_edit_k);
}
bool  color_plugin::inverted() const
{
  return checkbox_invert_k->isChecked();
}
}
