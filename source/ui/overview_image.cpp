#include /* implements */ <ui/overview_image.hpp>

#include <QMouseEvent>

namespace pli
{
overview_image::overview_image(QWidget* parent) : QLabel(parent), selection_square_(new selection_square(this))
{
  selection_square_->setMouseTracking(true);
}

void overview_image::set_selection_offset_percentage(const std::array<float, 2>& perc)
{
  selection_square_->move  (perc[0] * width(), perc[1] * height());
}
void overview_image::set_selection_size_percentage  (const std::array<float, 2>& perc)
{
  selection_square_->resize(perc[0] * width(), perc[1] * height());
}

std::array<float, 2>  overview_image::selection_offset_percentage() const
{
  auto val = selection_square_->pos();
  return {static_cast<float>(val.x    ()) / width(), static_cast<float>(val.y     ()) / height()};
}
std::array<float, 2>  overview_image::selection_size_percentage  () const
{
  auto val = selection_square_->size();
  return {static_cast<float>(val.width()) / width(), static_cast<float>(val.height()) / height()};
}

void overview_image::mousePressEvent(QMouseEvent* event)
{
  auto position = event->localPos();
  selection_square_->resize(selection_square_->minimumSize());
  selection_square_->move(position.x(), position.y());
  dragging_ = true;
  on_selection_change(selection_offset_percentage(), selection_size_percentage());
}
void overview_image::mouseReleaseEvent(QMouseEvent* event)
{
  dragging_ = false;
  on_selection_change(selection_offset_percentage(), selection_size_percentage());
}
void overview_image::mouseMoveEvent(QMouseEvent* event)
{
  if (dragging_)
  {
    auto size = event->localPos() - selection_square_->pos();
    selection_square_->resize(size.x(), size.y());
    on_selection_change(selection_offset_percentage(), selection_size_percentage());
  }
}
}
