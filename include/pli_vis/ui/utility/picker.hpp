#ifndef PLI_VIS_PICKER_
#define PLI_VIS_PICKER_

#include <QObject>

class QCustomEvent;
class QPoint;

class QwtPlot;
class QwtPlotCurve;

namespace pli
{
class picker : public QObject
{
  Q_OBJECT

public:
  picker(QwtPlot* plot);

  virtual bool eventFilter(QObject* sender, QEvent* event);
  virtual bool event      (QEvent* event);

signals:
  void on_change();

private:
  void select_or_add     (const QPoint& point);
  void remove            (const QPoint& point);
  void move              (const QPoint& point);
  void move_by           (int x, int y);

  void show_cursor       (bool enable);
  void shift_point_cursor(bool up    );
  void shift_curve_cursor(bool up    );

        QwtPlot* plot();
  const QwtPlot* plot() const;

  QwtPlotCurve* selected_curve_;
  int           selected_point_;
};
}

#endif
