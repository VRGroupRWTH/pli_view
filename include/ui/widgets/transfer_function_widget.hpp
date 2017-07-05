#ifndef PLI_VIS_TRANSFER_FUNCTION_WIDGET_
#define PLI_VIS_TRANSFER_FUNCTION_WIDGET_

#include <qwt/qwt_plot.h>

class QwtPlotCurve;
class QwtPlotHistogram;

namespace pli
{
class transfer_function_widget : public QwtPlot
{
public:
  transfer_function_widget(QWidget* parent = nullptr);

  void set_histogram_entries(const std::vector<std::size_t>& histogram_entries);

protected:
  QwtPlotCurve*     curves_[4];
  QwtPlotHistogram* histogram_;
};
}

#endif
