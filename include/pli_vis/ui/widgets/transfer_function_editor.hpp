#ifndef PLI_VIS_TRANSFER_FUNCTION_WIDGET_
#define PLI_VIS_TRANSFER_FUNCTION_WIDGET_

#include <array>

#include <qwt/qwt_plot.h>
#include <vector_types.h>

class QwtPlotCurve;
class QwtPlotHistogram;

namespace pli
{
class transfer_function_editor : public QwtPlot
{
  Q_OBJECT
  
public:
  transfer_function_editor(QWidget* parent = nullptr);

  std::vector<float4> get_function();

  void set_histogram_entries(const std::vector<std::size_t>& histogram_entries);

signals:
  void on_change();

protected:
  std::array<QwtPlotCurve*, 4> curves_   ;
  QwtPlotHistogram*            histogram_;
};
}

#endif
