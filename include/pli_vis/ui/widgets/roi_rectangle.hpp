#ifndef PLI_VIS_SELECTION_SQUARE_HPP_
#define PLI_VIS_SELECTION_SQUARE_HPP_

#include <memory>

#include <QRubberBand>
#include <QWidget>

namespace pli
{
class roi_rectangle : public QWidget
{
public:
  roi_rectangle(QWidget* parent = nullptr);

private:
  void resizeEvent(QResizeEvent* event) override;

  std::unique_ptr<QRubberBand> rubber_band_;
};
}

#endif
