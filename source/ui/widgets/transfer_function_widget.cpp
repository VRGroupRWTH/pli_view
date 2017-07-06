#include /* implements */ <ui/widgets/transfer_function_widget.hpp>

#include <third_party/qwt/qwt_curve_fitter.h>
#include <third_party/qwt/qwt_plot_curve.h>
#include <third_party/qwt/qwt_plot_grid.h>
#include <third_party/qwt/qwt_plot_histogram.h>
#include <third_party/qwt/qwt_symbol.h>

#include <ui/widgets/picker.hpp>

namespace pli
{
transfer_function_widget::transfer_function_widget(QWidget* parent) : QwtPlot(parent)
{
  auto font = axisFont(0);
  font.setPointSize(8);
  setAxisFont  (0, font);
  setAxisFont  (2, font);
  setAxisScale (0, 0, 255);
  setAxisScale (2, 0, 255);
  setAutoReplot(true);

  auto grid = new QwtPlotGrid;
  grid->enableXMin (true);
  grid->enableYMin (true);
  grid->setMajorPen(QPen(Qt::black, 0, Qt::DotLine));
  grid->setMinorPen(QPen(Qt::gray , 0, Qt::DotLine));
  grid->attach     (this);

  histogram_ = new QwtPlotHistogram;
  histogram_->setPen  (QPen  (Qt::gray, 0));
  histogram_->setBrush(QBrush(Qt::gray));
  histogram_->setStyle(QwtPlotHistogram::HistogramStyle::Columns);
  histogram_->attach  (this);

  for (auto i = 0; i < 4; i++)
  {
    curves_[i] = new QwtPlotCurve();
    curves_[i]->setStyle         (QwtPlotCurve::CurveStyle::Lines);
    curves_[i]->setCurveAttribute(QwtPlotCurve::CurveAttribute::Fitted);
    curves_[i]->setRenderHint    (QwtPlotItem::RenderHint::RenderAntialiased);
    curves_[i]->attach           (this);

    auto curve_fitter = new QwtSplineCurveFitter;
    curve_fitter->setFitMode    (QwtSplineCurveFitter::FitMode::ParametricSpline);
    curve_fitter->setSplineSize (100);
    curves_[i]  ->setCurveFitter(curve_fitter);

    // TESTING.
    QVector<QPointF> values;
    values.push_back(QPointF(0  , 0));
    values.push_back(QPointF(50 , rand() % 255));
    values.push_back(QPointF(100, rand() % 255));
    values.push_back(QPointF(150, rand() % 255));
    values.push_back(QPointF(200, rand() % 255));
    values.push_back(QPointF(250, 0));
    curves_[i]->setSamples(values); // fitter.fitCurve(values)
  }
  curves_[0]->setPen   (Qt::red);
  curves_[0]->setSymbol(new QwtSymbol(QwtSymbol::Ellipse, QBrush(Qt::red  ), QPen(Qt::red  , 1), QSize(4, 4)));
  curves_[1]->setPen   (Qt::green);
  curves_[1]->setSymbol(new QwtSymbol(QwtSymbol::Ellipse, QBrush(Qt::green), QPen(Qt::green, 1), QSize(4, 4)));
  curves_[2]->setPen   (Qt::blue );
  curves_[2]->setSymbol(new QwtSymbol(QwtSymbol::Ellipse, QBrush(Qt::blue ), QPen(Qt::blue , 1), QSize(4, 4)));
  curves_[3]->setPen   (Qt::black);
  curves_[3]->setSymbol(new QwtSymbol(QwtSymbol::Ellipse, QBrush(Qt::black), QPen(Qt::black, 1), QSize(4, 4)));

  new picker(this);
}

std::vector<float4> transfer_function_widget::get_function()
{
  std::vector<float4> function(256, float4{0.0, 0.0, 0.0, 0.0});
  for (auto i = 0; i < 125; i++)
    function[i] = float4{ float(i * 2) / 255.0F, float(i * 2) / 255.0F, float(i * 2) / 255.0F, float(i * 2) / 255.0F };
  return function;
}

void transfer_function_widget::set_curve            (std::size_t index, const std::vector<std::size_t>& curve)
{
  QVector<QPointF> samples;
  for (auto i = 0; i < curve.size(); i++)
    samples.push_back(QPointF(i, curve[i]));
  curves_[index]->setSamples(samples);
}
void transfer_function_widget::set_histogram_entries(const std::vector<std::size_t>& histogram_entries)
{
  QVector<QwtIntervalSample> samples;
  for (auto i = 0; i < histogram_entries.size(); i++)
    samples.push_back(QwtIntervalSample(histogram_entries[i], i, i + 1));
  histogram_->setSamples(samples);
}
}
