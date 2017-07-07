#include <pli_vis/ui/widgets/transfer_function_editor.hpp>

#include <qwt/qwt_curve_fitter.h>
#include <qwt/qwt_plot_curve.h>
#include <qwt/qwt_plot_grid.h>
#include <qwt/qwt_plot_histogram.h>
#include <qwt/qwt_symbol.h>

#include <pli_vis/ui/utility/plot_interactor.hpp>

namespace pli
{
transfer_function_editor::transfer_function_editor(QWidget* parent) : QwtPlot(parent)
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

    QVector<QPointF> values;
    values.push_back(QPointF(0  , 0     ));
    values.push_back(QPointF(125, 10 * i));
    values.push_back(QPointF(250, 0     ));
    curves_[i]->setSamples(values); 
  }
  curves_[0]->setPen   (Qt::red);
  curves_[0]->setSymbol(new QwtSymbol(QwtSymbol::Ellipse, QBrush(Qt::red  ), QPen(Qt::red  , 1), QSize(4, 4)));
  curves_[1]->setPen   (Qt::green);
  curves_[1]->setSymbol(new QwtSymbol(QwtSymbol::Ellipse, QBrush(Qt::green), QPen(Qt::green, 1), QSize(4, 4)));
  curves_[2]->setPen   (Qt::blue );
  curves_[2]->setSymbol(new QwtSymbol(QwtSymbol::Ellipse, QBrush(Qt::blue ), QPen(Qt::blue , 1), QSize(4, 4)));
  curves_[3]->setPen   (Qt::black);
  curves_[3]->setSymbol(new QwtSymbol(QwtSymbol::Ellipse, QBrush(Qt::black), QPen(Qt::black, 1), QSize(4, 4)));

  auto point_picker = new plot_interactor(this);
  connect(point_picker, SIGNAL(on_change()), this, SIGNAL(on_change()));
}

std::vector<float4> transfer_function_editor::get_function()
{
  std::vector<float4> function(256, float4{0.0, 0.0, 0.0, 0.0});
  QwtSplineCurveFitter fitter;
  fitter.setFitMode   (QwtSplineCurveFitter::FitMode::ParametricSpline);
  fitter.setSplineSize(256);
  QVector<QPointF> red_points  ; for (auto i = 0; i < curves_[0]->dataSize(); i++) red_points  .push_back(curves_[0]->sample(i)); auto reds   = fitter.fitCurve(red_points  );
  QVector<QPointF> green_points; for (auto i = 0; i < curves_[1]->dataSize(); i++) green_points.push_back(curves_[1]->sample(i)); auto greens = fitter.fitCurve(green_points);
  QVector<QPointF> blue_points ; for (auto i = 0; i < curves_[2]->dataSize(); i++) blue_points .push_back(curves_[2]->sample(i)); auto blues  = fitter.fitCurve(blue_points );
  QVector<QPointF> alpha_points; for (auto i = 0; i < curves_[3]->dataSize(); i++) alpha_points.push_back(curves_[3]->sample(i)); auto alphas = fitter.fitCurve(alpha_points);
  for (auto i = 0; i < 256; i++)
    function[i] = float4{float(reds[i].y()) / 255.0F, float(greens[i].y()) / 255.0F, float(blues[i].y()) / 255.0F, float(alphas[i].y()) / 255.0F};
  return function;
}

void transfer_function_editor::set_histogram_entries(const std::vector<std::size_t>& histogram_entries)
{
  QVector<QwtIntervalSample> samples;
  for (auto i = 0; i < histogram_entries.size(); i++)
    samples.push_back(QwtIntervalSample(histogram_entries[i], i, i + 1));
  histogram_->setSamples(samples);
}
}
