#include <pli_vis/ui/utility/plot_interactor.hpp>

#include <qapplication.h>
#include <qevent.h>
#include <qwhatsthis.h>
#include <qpainter.h>

#include <qwt/qwt_plot.h>
#include <qwt/qwt_symbol.h>
#include <qwt/qwt_plot_canvas.h>
#include <qwt/qwt_plot_curve.h>
#include <qwt/qwt_plot_directpainter.h>

namespace pli
{
plot_interactor::plot_interactor(QwtPlot* plot) : QObject(plot), selected_curve_(nullptr), selected_point_(-1)
{
  auto canvas = qobject_cast<QwtPlotCanvas*>(plot->canvas());
  canvas->installEventFilter(this);

  canvas->setFocusPolicy   (Qt::StrongFocus);
  canvas->setCursor        (Qt::PointingHandCursor);
  canvas->setFocusIndicator(QwtPlotCanvas::ItemFocusIndicator);
  canvas->setFocus         ();

  shift_curve_cursor(true);
}

bool plot_interactor::eventFilter(QObject* object, QEvent* event)
{
  if (plot() == nullptr || object != plot()->canvas())
    return false;

  switch (event->type())
  {
  case QEvent::FocusIn:
  {
    show_cursor(true);
    break;
  }
  case QEvent::FocusOut:
  {
    show_cursor(false);
    break;
  }
  case QEvent::Paint:
  {
    QApplication::postEvent(this, new QEvent(QEvent::User));
    break;
  }
  case QEvent::MouseButtonPress:
  {
    const QMouseEvent* mouse_event = static_cast<QMouseEvent *>(event);

    if (mouse_event->buttons() == Qt::LeftButton)
      select_or_add(mouse_event->pos());
    else if (mouse_event->buttons() == Qt::RightButton)
      remove(mouse_event->pos());
    return true;
  }
  case QEvent::MouseMove:
  {
    const QMouseEvent* mouse_event = static_cast<QMouseEvent*>(event);
    move(mouse_event->pos());
    return true;
  }
  case QEvent::KeyPress:
  {
    const QKeyEvent* key_event = static_cast<QKeyEvent*>(event);

    const auto delta = 5;
    switch (key_event->key())
    {
    case Qt::Key_Up:
    {
      shift_curve_cursor(true);
      return true;
    }
    case Qt::Key_Down:
    {
      shift_curve_cursor(false);
      return true;
    }
    case Qt::Key_Right:
    case Qt::Key_Plus:
    {
      if (selected_curve_)
        shift_point_cursor(true);
      else
        shift_curve_cursor(true);
      return true;
    }
    case Qt::Key_Left:
    case Qt::Key_Minus:
    {
      if (selected_curve_)
        shift_point_cursor(false);
      else
        shift_curve_cursor(true);
      return true;
    }

    case Qt::Key_1:
    {
      move_by(-delta, delta);
      break;
    }
    case Qt::Key_2:
    {
      move_by(0, delta);
      break;
    }
    case Qt::Key_3:
    {
      move_by(delta, delta);
      break;
    }
    case Qt::Key_4:
    {
      move_by(-delta, 0);
      break;
    }
    case Qt::Key_6:
    {
      move_by(delta, 0);
      break;
    }
    case Qt::Key_7:
    {
      move_by(-delta, -delta);
      break;
    }
    case Qt::Key_8:
    {
      move_by(0, -delta);
      break;
    }
    case Qt::Key_9:
    {
      move_by(delta, -delta);
      break;
    }
    default:
      break;
    }
  }
  default:
    break;
  }

  return QObject::eventFilter(object, event);
}
bool plot_interactor::event      (QEvent*  event)
{
  if (event->type() == QEvent::User)
  {
    show_cursor(true);
    return true;
  }
  return QObject::event(event);
}

void plot_interactor::select_or_add(const QPoint& pos)
{
  QwtPlotCurve* curve = nullptr;
  auto dist  = 10e10;
  auto index = -1;

  const auto& item_list = plot()->itemList();
  for (auto it = item_list.begin(); it != item_list.end(); ++it)
  {
    if ((*it)->rtti() == QwtPlotItem::Rtti_PlotCurve)
    {
      auto c = static_cast<QwtPlotCurve*>(*it);
      
      double d;
      auto idx = c->closestPoint(pos, &d);
      if (d < dist && idx != 0 && idx != c->dataSize() - 1)
      {
        curve = c;
        index = idx;
        dist  = d;
      }
    }
  }

  show_cursor(false);
  selected_curve_ = nullptr;
  selected_point_ = -1;

  if (curve)
  {
    selected_curve_ = curve;
    selected_point_ = index;
    show_cursor(true);

    if(dist >= 8)
    {
      auto x = plot()->invTransform(selected_curve_->xAxis(), pos.x());
      auto y = plot()->invTransform(selected_curve_->yAxis(), pos.y());
      QPointF sample(x, y);

      selected_point_ = curve->sample(index).x() < sample.x() ? index + 1 : index;

      QVector<QPointF> samples;
      for (auto i = 0; i < curve->dataSize(); ++i)
        samples.push_back(curve->sample(i));
      samples.insert(selected_point_, sample);
      curve->setSamples(samples);
    }
  }

  on_change();
}
void plot_interactor::remove       (const QPoint& pos)
{
  QwtPlotCurve* curve = nullptr;
  auto dist  = 10e10;
  auto index = -1;

  const auto& item_list = plot()->itemList();
  for (auto it = item_list.begin(); it != item_list.end(); ++it)
  {
    if ((*it)->rtti() == QwtPlotItem::Rtti_PlotCurve)
    {
      auto c = static_cast<QwtPlotCurve*>(*it);

      double d;
      auto idx = c->closestPoint(pos, &d);
      if (d < dist && idx != 0 && idx != c->dataSize() - 1)
      {
        curve = c;
        index = idx;
        dist  = d;
      }
    }
  }

  show_cursor(false);
  selected_curve_ = nullptr;
  selected_point_ = -1;

  if (curve && dist < 4)
  {
    QVector<QPointF> samples;
    for(auto i = 0; i < curve->dataSize(); ++i)
      if(i != index)
        samples.push_back(curve->sample(i));
    curve->setSamples(samples);
  }

  on_change();
}
void plot_interactor::move         (const QPoint& pos)
{
  if (!selected_curve_)
    return;

  QVector<double> xData(selected_curve_->dataSize());
  QVector<double> yData(selected_curve_->dataSize());

  for (auto i = 0; i < static_cast<int>(selected_curve_->dataSize()); i++)
  {
    if (i == selected_point_)
    {
      xData[i] = plot()->invTransform(selected_curve_->xAxis(), pos.x());
      yData[i] = plot()->invTransform(selected_curve_->yAxis(), pos.y());
    }
    else
    {
      const auto sample = selected_curve_->sample(i);
      xData[i] = sample.x();
      yData[i] = sample.y();
    }
  }
  selected_curve_->setSamples(xData, yData);

  auto plotCanvas = qobject_cast<QwtPlotCanvas *>(plot()->canvas());

  plotCanvas->setPaintAttribute(QwtPlotCanvas::ImmediatePaint, true);
  plot()->replot();
  plotCanvas->setPaintAttribute(QwtPlotCanvas::ImmediatePaint, false);

  show_cursor(true);

  on_change();
}
void plot_interactor::move_by      (int dx, int dy)
{
  if (dx == 0 && dy == 0)
    return;
  if (!selected_curve_)
    return;
  const auto sample = selected_curve_->sample(selected_point_);
  const auto x      = plot()->transform(selected_curve_->xAxis(), sample.x());
  const auto y      = plot()->transform(selected_curve_->yAxis(), sample.y());
  move(QPoint(qRound(x + dx), qRound(y + dy)));

  on_change();
}

void plot_interactor::show_cursor       (bool enable)
{
  if (!selected_curve_)
    return;

  auto symbol = const_cast<QwtSymbol *>(selected_curve_->symbol());

  const auto brush = symbol->brush();
  if (enable)
    symbol->setBrush(symbol->brush().color().dark(180));

  QwtPlotDirectPainter direct_painter;
  direct_painter.drawSeries(selected_curve_, selected_point_, selected_point_);

  if (enable)
    symbol->setBrush(brush);
}
void plot_interactor::shift_curve_cursor(bool up    )
{
  QwtPlotItemIterator it;

  const auto& item_list = plot()->itemList();

  QwtPlotItemList curve_list;
  for (it = item_list.begin(); it != item_list.end(); ++it)
  {
    if ((*it)->rtti() == QwtPlotItem::Rtti_PlotCurve)
      curve_list += *it;
  }
  if (curve_list.isEmpty())
    return;

  it = curve_list.begin();

  if (selected_curve_)
  {
    for (it = curve_list.begin(); it != curve_list.end(); ++it)
    {
      if (selected_curve_ == *it)
        break;
    }
    if (it == curve_list.end())
      it = curve_list.begin();

    if (up)
    {
      ++it;
      if (it == curve_list.end())
        it = curve_list.begin();
    }
    else
    {
      if (it == curve_list.begin())
        it = curve_list.end();
      --it;
    }
  }

  show_cursor(false);
  selected_point_ = 0;
  selected_curve_ = static_cast<QwtPlotCurve *>(*it);
  show_cursor(true );
}
void plot_interactor::shift_point_cursor(bool up    )
{
  if (!selected_curve_)
    return;

  auto index = selected_point_ + (up ? 1 : -1);
  index = (index + selected_curve_->dataSize()) % selected_curve_->dataSize();

  if (index != selected_point_)
  {
    show_cursor(false);
    selected_point_ = index;
    show_cursor(true);
  }
}
  
      QwtPlot* plot_interactor::plot()
{
  return qobject_cast<QwtPlot*>(parent());
}
const QwtPlot* plot_interactor::plot() const
{
  return qobject_cast<const QwtPlot *>(parent());
}

}