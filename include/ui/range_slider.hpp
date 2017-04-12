#ifndef PLI_VIS_RANGE_SLIDER_HPP_
#define PLI_VIS_RANGE_SLIDER_HPP_

#include <QSlider>

namespace pli
{
class range_slider : public QSlider
{
  Q_OBJECT

public:
  range_slider(QWidget* parent = nullptr);

private:

};
}

#endif
