#include /* implements */ <ui/toolboxes/fom_toolbox_widget.hpp>

#include <limits>

namespace pli
{
fom_toolbox_widget::fom_toolbox_widget(QWidget* parent)
: QWidget(parent)
{
  setupUi(this);

  line_edit_scale->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));
}
}