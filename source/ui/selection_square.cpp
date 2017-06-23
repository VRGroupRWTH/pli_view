#include /* implements */ <ui/selection_square.hpp>

#include <QHBoxLayout>
#include <QSizeGrip>

namespace pli
{
selection_square::selection_square(QWidget* parent) : QWidget(parent), rubber_band_(new QRubberBand(QRubberBand::Rectangle, this))
{
  setWindowFlags(Qt::SubWindow);
  
  auto layout = new QHBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->addWidget(new QSizeGrip(this), 0, Qt::AlignLeft  | Qt::AlignTop   );
  layout->addWidget(new QSizeGrip(this), 0, Qt::AlignRight | Qt::AlignBottom);

  auto inner_palette = rubber_band_->palette();
  inner_palette.setColor(QPalette::Highlight, Qt::darkRed);
  rubber_band_->setPalette(inner_palette);
  rubber_band_->show();

  setMinimumSize(1, 1);
  move(0, 0);
  show();
}
void selection_square::resizeEvent(QResizeEvent* event)
{
  rubber_band_->resize(size());
}
}
