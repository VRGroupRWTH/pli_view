#include "pli_vis/ui/widgets/wait_spinner.hpp"

#include <cmath>
#include <algorithm>

#include <QPainter>
#include <QTimer>

namespace pli
{
wait_spinner::wait_spinner(QWidget* parent, bool center, bool disable_parent)
: QWidget      (parent),
  center_        (center),
  disable_parent_(disable_parent) 
{
  initialize();
}
wait_spinner::wait_spinner(Qt::WindowModality modality, QWidget* parent, bool center, bool disable_parent)
: QWidget        (parent, Qt::Dialog | Qt::FramelessWindowHint),
  center_        (center),
  disable_parent_(disable_parent)
{
  initialize();

  setWindowModality(modality);
  setAttribute     (Qt::WA_TranslucentBackground);
}

void   wait_spinner::paintEvent                (QPaintEvent* paint_event) 
{
  update_position();

  QPainter painter(this);
  painter.fillRect(this->rect(), Qt::transparent);
  painter.setRenderHint(QPainter::Antialiasing, true);

  if (current_counter_ >= number_of_lines_)
    current_counter_ = 0;

  painter.setPen(Qt::NoPen);
  for (auto i = 0; i < number_of_lines_; ++i) 
  {
    painter.save();
    painter.translate(inner_radius_ + line_length_, inner_radius_ + line_length_);

    auto angle = static_cast<qreal>(360 * i) / static_cast<qreal>(number_of_lines_);

    painter.rotate   (angle);
    painter.translate(inner_radius_, 0);

    auto distance = line_count_distance(i, current_counter_, number_of_lines_);
    auto color    = calculate_color(distance, number_of_lines_, trail_fade_, minimum_trail_opacity_, color_);
    painter.setBrush(color);

    painter.drawRoundedRect(QRect(0, -line_width_ / 2, line_length_, line_width_), roundness_, roundness_, Qt::RelativeSize);
    painter.restore();
  }
}
       
void   wait_spinner::start                     () 
{
  update_position();
  is_spinning_ = true;
  show();

  if (parentWidget() && disable_parent_)
    parentWidget()->setEnabled(false);

  if (!timer_->isActive()) {
    timer_->start();
    current_counter_ = 0;
  }
}
void   wait_spinner::stop                      () 
{
  is_spinning_ = false;
  hide();

  if (parentWidget() && disable_parent_)
    parentWidget()->setEnabled(true);

  if (timer_->isActive()) {
    timer_->stop();
    current_counter_ = 0;
  }
}
       
void   wait_spinner::set_color                 (QColor color                 ) 
{
  color_ = color;
}
void   wait_spinner::set_roundness             (qreal  roundness             ) 
{
  roundness_ = std::max(0.0, std::min(100.0, roundness));
}
void   wait_spinner::set_minimum_trail_opacity (qreal  minimum_trail_opacity ) 
{
  minimum_trail_opacity_ = minimum_trail_opacity;
}
void   wait_spinner::set_trail_fade            (qreal  trail_fade            ) 
{
  trail_fade_ = trail_fade;
}
void   wait_spinner::set_revolutions_per_second(qreal  revolutions_per_second) 
{
  revolutions_per_second_ = revolutions_per_second;
  update_timer();
}
void   wait_spinner::set_number_of_lines       (int    lines                 )
{
  number_of_lines_ = lines;
  current_counter_ = 0;
  update_timer();
}
void   wait_spinner::set_line_length           (int    length                ) 
{
  line_length_ = length;
  update_size();
}
void   wait_spinner::set_line_width            (int    width                 ) 
{
  line_width_ = width;
  update_size();
}
void   wait_spinner::set_inner_radius          (int    radius                ) 
{
  inner_radius_ = radius;
  update_size();
}

QColor wait_spinner::color                     () const
{
  return color_;
}
qreal  wait_spinner::roundness                 () const
{
  return roundness_;
}
qreal  wait_spinner::minimum_trail_opacity     () const
{
  return minimum_trail_opacity_;
}
qreal  wait_spinner::trail_fade                () const
{
  return trail_fade_;
}
qreal  wait_spinner::revolutions_pers_second   () const
{
  return revolutions_per_second_;
}
int    wait_spinner::number_of_lines           () const
{
  return number_of_lines_;
}
int    wait_spinner::line_length               () const
{
  return line_length_;
}
int    wait_spinner::line_width                () const
{
  return line_width_;
}
int    wait_spinner::inner_radius              () const
{
  return inner_radius_;
}
                                               
bool   wait_spinner::is_spinning               () const
{
  return is_spinning_;
}
                                               
void   wait_spinner::rotate                    () 
{
  ++current_counter_;
  if (current_counter_ >= number_of_lines_)
    current_counter_ = 0;
  update();
}       
                                               
void   wait_spinner::initialize                () 
{
  timer_ = new QTimer(this);
  connect(timer_, SIGNAL(timeout()), this, SLOT(rotate()));

  update_size ();
  update_timer();
  hide        ();
}
void   wait_spinner::update_size               () 
{
  auto size = (inner_radius_ + line_length_) * 2;
  setFixedSize(size, size);
}
void   wait_spinner::update_timer              ()
{
  timer_->setInterval(1000 / (number_of_lines_ * revolutions_per_second_));
}
void   wait_spinner::update_position           () 
{
  if (parentWidget() && center_)
    move(parentWidget()->width() / 2 - width() / 2, parentWidget()->height() / 2 - height() / 2);
}
                                               
int    wait_spinner::line_count_distance       (int current, int primary, int line_count) 
{
  auto distance = primary - current;
  if (distance < 0)
    distance += line_count;
  return distance;
}
QColor wait_spinner::calculate_color           (int count_distance, int line_count, qreal trail_fade, qreal min_opacity, QColor color) {
  if (count_distance == 0)
    return color;
  const auto min_alpha = min_opacity / 100.0;
        auto threshold = static_cast<int>(ceil((line_count - 1) * trail_fade / 100.0));
  if (count_distance > threshold)
    color.setAlphaF(min_alpha);
  else 
  {
    auto difference   = color.alphaF() - min_alpha;
    auto gradient     = difference / static_cast<qreal>(threshold + 1);
    auto result_alpha = color.alphaF() - gradient * count_distance;

    result_alpha = std::min(1.0, std::max(0.0, result_alpha));
    color.setAlphaF(result_alpha);
  }
  return color;
}
}