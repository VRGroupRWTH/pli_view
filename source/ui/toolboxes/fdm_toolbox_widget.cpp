#include /* implements */ <ui/toolboxes/fdm_toolbox_widget.hpp>

#include <limits>

namespace pli
{
fdm_toolbox_widget::fdm_toolbox_widget(QWidget* parent)
: QWidget(parent)
{
  setupUi(this);
         
  line_edit_block_size_x    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_block_size_y    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_block_size_z    ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_histogram_bins_x->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_histogram_bins_y->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_max_order       ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_samples_x       ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_samples_y       ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
}
}