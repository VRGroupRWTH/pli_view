#include /* implements */ <ui/span_slider.hpp>

#include <QKeyEvent>
#include <QMouseEvent>
#include <QApplication>
#include <QStylePainter>
#include <QStyleOptionSlider>

span_slider::span_slider(                             QWidget* parent) : QSlider(parent),
  first_movement_(false),
  block_tracking_(false),
  lower_         (0),
  upper_         (0),
  lower_position_(0),
  upper_position_(0),
  offset_        (0),
  position_      (0),
  last_pressed_  (NoHandle       ),
  main_control_  (LowerHandle    ),
  movement_mode_ (FreeMovement   ),
  lower_pressed_ (QStyle::SC_None),
  upper_pressed_ (QStyle::SC_None)
{
  connect(this, SIGNAL(sliderReleased()), this, SLOT(movePressedHandle()));
}
span_slider::span_slider(Qt::Orientation orientation, QWidget* parent) : QSlider(orientation, parent)
{
  connect(this, SIGNAL(sliderReleased()), this, SLOT(movePressedHandle()));
}

span_slider::handle_movement_mode span_slider::movement_mode      () const
{
  return movement_mode_;
}
void                              span_slider::set_movement_mode  (handle_movement_mode mode)
{
  movement_mode_ = mode;
}
int                               span_slider::lower_value        () const
{
  return qMin(lower_, upper_);
}
void                              span_slider::set_lower_value    (int lower)
{
  set_span(lower, upper_);
}
int                               span_slider::upper_value        () const
{
  return qMax(lower_, upper_);
}
void                              span_slider::set_upper_value    (int upper)
{
  set_span(lower_, upper);
}
int                               span_slider::lower_position     () const
{
  return lower_position_;
}
void                              span_slider::set_lower_position (int lower)
{
  if (lower_position_ != lower)
  {
    lower_position_ = lower;
    if (!hasTracking())
      update();
    if (isSliderDown())
      emit lower_position_changed(lower);
    if (hasTracking() && !block_tracking_)
    {
      trigger_action(SliderMove, main_control_ == LowerHandle);
    }
  }
}
int                               span_slider::upper_position     () const
{
  return upper_position_;
}
void                              span_slider::set_upper_position (int upper)
{
  if (upper_position_ != upper)
  {
    upper_position_ = upper;
    if (!hasTracking())
      update();
    if (isSliderDown())
      emit upper_position_changed(upper);
    if (hasTracking() && !block_tracking_)
    {
      bool main = (main_control_ == UpperHandle);
      trigger_action(SliderMove, main);
    }
  }
}
                                                                  
void                              span_slider::set_span           (int lower, int upper)
{
  const auto low = qBound(minimum(), qMin(lower, upper), maximum());
  const auto upp = qBound(minimum(), qMax(lower, upper), maximum());
  if (low != lower || upp != upper)
  {
    if (low != lower)
    {
      lower           = low;
      lower_position_ = low;
      emit lower_value_changed(low);
    }
    if (upp != upper)
    {
      upper           = upp;
      upper_position_ = upp;
      emit upper_value_changed(upp);
    }

    update_range(lower, upper);
    emit span_changed(lower, upper);
    update();
  }
}
void                              span_slider::update_range       (int min  , int max  )
{
  Q_UNUSED(min);
  Q_UNUSED(max);
  set_span(lower_, upper_);
}
void                              span_slider::move_pressed_handle()
{
  switch (last_pressed_)
  {
  case LowerHandle:
    if (lower_position_ != lower_)
      trigger_action(SliderMove, main_control_ == LowerHandle);
    break;
  case UpperHandle:
    if (upper_position_ != upper_)
      trigger_action(SliderMove, main_control_ == UpperHandle);
    break;
  default:
    break;
  }
}

void                              span_slider::keyPressEvent      (QKeyEvent* event)
{
  QSlider::keyPressEvent(event);

  auto main   = true;
  auto action = SliderNoAction;
  switch (event->key())
  {
  case Qt::Key_Left:
    main   = orientation() == Qt::Horizontal;
    action = !invertedAppearance() ? SliderSingleStepSub : SliderSingleStepAdd;
    break;
  case Qt::Key_Right:
    main   = orientation() == Qt::Horizontal;
    action = !invertedAppearance() ? SliderSingleStepAdd : SliderSingleStepSub;
    break;
  case Qt::Key_Up:
    main   = orientation() == Qt::Vertical;
    action = invertedControls() ? SliderSingleStepSub : SliderSingleStepAdd;
    break;
  case Qt::Key_Down:
    main   = orientation() == Qt::Vertical;
    action = invertedControls() ? SliderSingleStepAdd : SliderSingleStepSub;
    break;
  case Qt::Key_Home:
    main   = main_control_ == LowerHandle;
    action = SliderToMinimum;
    break;
  case Qt::Key_End:
    main   = main_control_ == UpperHandle;
    action = SliderToMaximum;
    break;
  default:
    event->ignore();
    break;
  }

  if (action)
    trigger_action(action, main);
}
void                              span_slider::mousePressEvent    (QMouseEvent* event)
{
  if (minimum() == maximum() || (event->buttons() ^ event->button()))
  {
    event->ignore();
    return;
  }

  handle_mouse_press(event->pos(), upper_pressed_, upper_, UpperHandle);

  if (upper_pressed_ != QStyle::SC_SliderHandle)
    handle_mouse_press(event->pos(), lower_pressed_, lower_, LowerHandle);

  first_movement_ = true;
  event->accept();
}
void                              span_slider::mouseMoveEvent     (QMouseEvent* event)
{
  if (lower_pressed_ != QStyle::SC_SliderHandle && upper_pressed_ != QStyle::SC_SliderHandle)
  {
    event->ignore();
    return;
  }

  QStyleOptionSlider opt;
  init_style_option(&opt);
  const auto m = style()->pixelMetric(QStyle::PM_MaximumDragDistance, &opt, this);

  auto new_position = pixel_pos_to_range_value(pick(event->pos()) - offset_);
  if (m >= 0)
  {
    const auto r = rect().adjusted(-m, -m, m, m);
    if (!r.contains(event->pos()))
      new_position = position_;
  }

  if (first_movement_)
  {
    if (lower_ == upper_)
    {
      if (new_position < lower_value())
      {
        swap_controls();
        first_movement_ = false;
      }
    }
    else
      first_movement_ = false;
  }

  if (lower_pressed_ == QStyle::SC_SliderHandle)
  {
    if (movement_mode_ == NoCrossing)
      new_position = qMin(new_position, upper_value());
    else if (movement_mode_ == NoOverlapping)
      new_position = qMin(new_position, upper_value() - 1);

    if (movement_mode_ == FreeMovement && new_position > upper_)
    {
      swap_controls();
      set_upper_position(new_position);
    }
    else
      set_lower_position(new_position);
  }
  else if (upper_pressed_ == QStyle::SC_SliderHandle)
  {
    if (movement_mode_ == NoCrossing)
      new_position = qMax(new_position, lower_value());
    else if (movement_mode_ == NoOverlapping)
      new_position = qMax(new_position, lower_value() + 1);

    if (movement_mode_ == FreeMovement && new_position < lower_)
    {
      swap_controls();
      set_lower_position(new_position);
    }
    else
      set_upper_position(new_position);
  }
  event->accept();
}
void                              span_slider::mouseReleaseEvent  (QMouseEvent* event)
{
  QSlider::mouseReleaseEvent(event);
  setSliderDown(false);
  lower_pressed_ = QStyle::SC_None;
  upper_pressed_ = QStyle::SC_None;
  update();
}
void                              span_slider::paintEvent         (QPaintEvent* event)
{
  Q_UNUSED(event);
  QStylePainter painter(this);

  QStyleOptionSlider opt;
  init_style_option(&opt);
  opt.sliderValue    = 0;
  opt.sliderPosition = 0;
  opt.subControls    = QStyle::SC_SliderGroove | QStyle::SC_SliderTickmarks;
  painter.drawComplexControl(QStyle::CC_Slider, opt);

  opt.sliderPosition = lower_position_;
  const auto lr  = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, this);
  const auto lrv = pick(lr.center());
  opt.sliderPosition = upper_position_;
  const auto ur  = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, this);
  const auto urv = pick(ur.center());

  const auto minv = qMin(lrv, urv);
  const auto maxv = qMax(lrv, urv);
  const auto c    = QRect(lr.center(), ur.center()).center();
  auto span_rect  = orientation() == Qt::Horizontal ? QRect(QPoint(minv, c.y() - 2), QPoint(maxv, c.y() + 1)) : QRect(QPoint(c.x() - 2, minv), QPoint(c.x() + 1, maxv));
  draw_span(&painter, span_rect);

  switch (last_pressed_)
  {
  case LowerHandle:
    draw_handle(&painter, UpperHandle);
    draw_handle(&painter, LowerHandle);
    break;
  case UpperHandle:
  default:
    draw_handle(&painter, LowerHandle);
    draw_handle(&painter, UpperHandle);
    break;
  }
}

void span_slider::init_style_option       (QStyleOptionSlider* option, span_handle handle) const
{
  init_style_option(option);
  option->sliderPosition = handle == LowerHandle ? lower_position_ : upper_position_;
  option->sliderValue    = handle == LowerHandle ? lower_ : upper_;
}
int  span_slider::pick                    (const QPoint& point) const
{
  return orientation() == Qt::Horizontal ? point.x() : point.y();
}
int  span_slider::pixel_pos_to_range_value(int position) const
{
  QStyleOptionSlider opt;
  init_style_option(&opt);

  auto       slider_min    = 0;
  auto       slider_max    = 0;
  auto       slider_length = 0;
  const auto gr            = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderGroove, this);
  const auto sr            = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, this);
  if (orientation() == Qt::Horizontal)
  {
    slider_length = sr.width();
    slider_min    = gr.x    ();
    slider_max    = gr.right() - slider_length + 1;
  }
  else
  {
    slider_length = sr.height();
    slider_min    = gr.y     ();
    slider_max    = gr.bottom() - slider_length + 1;
  }
  return QStyle::sliderValueFromPosition(
    minimum(), maximum(), position - slider_min, 
    slider_max - slider_min, opt.upsideDown);
}
void span_slider::handle_mouse_press      (const QPoint& position, QStyle::SubControl& control, int value, span_handle handle)
{
  QStyleOptionSlider opt;
  init_style_option(&opt, handle);
  const auto old_control = control;
  control = style()->hitTestComplexControl(QStyle::CC_Slider, &opt, position, this);
  const auto sr = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, this);
  if (control == QStyle::SC_SliderHandle)
  {
    position_     = value;
    offset_       = pick(position - sr.topLeft());
    last_pressed_ = handle;
    setSliderDown(true);
    emit slider_pressed(handle);
  }
  if (control != old_control)
    update(sr);
}
void span_slider::setup_painter           (QPainter* painter, Qt::Orientation orientation, qreal x1, qreal y1, qreal x2, qreal y2) const
{
  auto highlight = palette().color(QPalette::Highlight);
  QLinearGradient gradient(x1, y1, x2, y2);
  gradient.setColorAt(0, highlight.dark (120));
  gradient.setColorAt(1, highlight.light(108));
  painter->setBrush(gradient);
  painter->setPen(QPen(highlight.dark(orientation == Qt::Horizontal ? 130 : 150), 0));
}
void span_slider::draw_span               (QStylePainter* painter, const QRect& rectangle) const
{
  QStyleOptionSlider opt;
  init_style_option(&opt);

  auto groove = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderGroove, this);
  opt.orientation == Qt::Horizontal ? groove.adjust(0, 0, -1, 0) : groove.adjust(0, 0, 0, -1);

  painter->setPen(QPen(palette().color(QPalette::Dark).light(110), 0));
  opt.orientation == Qt::Horizontal ? setup_painter(painter, opt.orientation, groove.center().x(), groove.top(), groove.center().x(), groove.bottom()) : setup_painter(painter, opt.orientation, groove.left(), groove.center().y(), groove.right(), groove.center().y());

  painter->drawRect(rectangle.intersected(groove));
}
void span_slider::draw_handle             (QStylePainter* painter, span_handle handle) const
{
  QStyleOptionSlider opt;
  init_style_option(&opt, handle);
  opt.subControls = QStyle::SC_SliderHandle;
  auto pressed = (handle == LowerHandle ? lower_pressed_ : upper_pressed_);
  if (pressed == QStyle::SC_SliderHandle)
  {
    opt.activeSubControls = pressed;
    opt.state |= QStyle::State_Sunken;
  }
  painter->drawComplexControl(QStyle::CC_Slider, opt);
}
void span_slider::trigger_action          (SliderAction action, bool main)
{
  auto       value       = 0;
  auto       no          = false;
  auto       up          = false;
  const auto min         = minimum();
  const auto max         = maximum();
  const auto alt_control = (main_control_ == LowerHandle ? UpperHandle : LowerHandle);

  block_tracking_ = true;

  switch (action)
  {
  case SliderSingleStepAdd:
    if (main && main_control_ == UpperHandle || (!main && alt_control == UpperHandle))
    {
      value = qBound(min, upper_ + singleStep(), max);
      up = true;
      break;
    }
    value = qBound(min, lower_ + singleStep(), max);
    break;
  case SliderSingleStepSub:
    if (main && main_control_ == UpperHandle || (!main && alt_control == UpperHandle))
    {
      value = qBound(min, upper_ - singleStep(), max);
      up = true;
      break;
    }
    value = qBound(min, lower_ - singleStep(), max);
    break;
  case SliderToMinimum:
    value = min;
    if (main && main_control_ == UpperHandle || (!main && alt_control == UpperHandle))
      up = true;
    break;
  case SliderToMaximum:
    value = max;
    if (main && main_control_ == UpperHandle || (!main && alt_control == UpperHandle))
      up = true;
    break;
  case SliderMove:
    if (main && main_control_ == UpperHandle || (!main && alt_control == UpperHandle))
      up = true;
  case SliderNoAction:
    no = true;
    break;
  default:
    break;
  }

  if (!no && !up)
  {
    if (movement_mode_ == NoCrossing)
      value = qMin(value, upper_);
    else if (movement_mode_ == NoOverlapping)
      value = qMin(value, upper_ - 1);

    if (movement_mode_ == FreeMovement && value > upper_)
    {
      swap_controls();
      set_upper_position(value);
    }
    else
      set_lower_position(value);
  }
  else if (!no)
  {
    if (movement_mode_ == NoCrossing)
      value = qMax(value, lower_);
    else if (movement_mode_ == NoOverlapping)
      value = qMax(value, lower_ + 1);

    if (movement_mode_ == FreeMovement && value < lower_)
    {
      swap_controls();
      set_lower_position(value);
    }
    else
      set_upper_position(value);
  }

  block_tracking_ = false;
  set_lower_value(lower_position_);
  set_upper_value(upper_position_);
}
void span_slider::swap_controls           ()
{
  qSwap(lower_, upper_);
  qSwap(lower_pressed_, upper_pressed_);
  last_pressed_ = last_pressed_ == LowerHandle ? UpperHandle : LowerHandle;
  main_control_ = main_control_ == LowerHandle ? UpperHandle : LowerHandle;
}
