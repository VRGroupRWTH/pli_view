#include /* implements */ <ui/widgets/transfer_function_widget.hpp>

#include <third_party/qwt/qwt_plot_curve.h>
#include <third_party/qwt/qwt_plot_grid.h>
#include <third_party/qwt/qwt_plot_histogram.h>

namespace pli
{
transfer_function_widget::transfer_function_widget(QWidget* parent) : QwtPlot(parent)
{
  // Setup transfer function widget.
  auto font = axisFont(0);
  font.setPointSize(8);
  setAxisFont     (0, font);
  setAxisFont     (2, font);
  setAxisAutoScale(0);
  setAxisScale    (2, 0, 255);
  setAutoReplot   (true);

  auto grid  = new QwtPlotGrid;
  histogram_ = new QwtPlotHistogram;
  grid      ->enableXMin (true);
  grid      ->enableYMin (true);
  grid      ->setMajorPen(QPen  (Qt::black, 0, Qt::DotLine  ));
  grid      ->setMinorPen(QPen  (Qt::gray , 0, Qt::DotLine  ));
  histogram_->setPen     (QPen  (Qt::gray , 0));
  histogram_->setBrush   (QBrush(Qt::gray));
  histogram_->setStyle   (QwtPlotHistogram::HistogramStyle::Columns);
  grid      ->attach     (this);
  histogram_->attach     (this);

  for (auto i = 0; i < 4; i++)
  {
    curves_[i] = new QwtPlotCurve();
    curves_[i]->setStyle(QwtPlotCurve::CurveStyle::Lines);
    curves_[i]->attach  (this);

    // TESTING.
    QVector<QPointF> values;
    values.push_back(QPointF(rand() % 255, rand() % 1000));
    values.push_back(QPointF(rand() % 255, rand() % 1000));
    values.push_back(QPointF(rand() % 255, rand() % 1000));
    values.push_back(QPointF(rand() % 255, rand() % 1000));
    curves_[i]->setSamples(values);
  }
  curves_[0]->setPen(Qt::red  );
  curves_[1]->setPen(Qt::green);
  curves_[2]->setPen(Qt::blue );
  curves_[3]->setPen(Qt::gray );
}

void transfer_function_widget::set_histogram_entries(const std::vector<std::size_t>& histogram_entries)
{
  QVector<QwtIntervalSample> samples;
  for (auto i = 0; i < histogram_entries.size(); i++)
    samples.push_back(QwtIntervalSample(histogram_entries[i], i, i + 1));
  histogram_->setSamples(samples);
}
}
