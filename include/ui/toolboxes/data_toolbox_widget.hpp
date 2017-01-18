#ifndef PLI_VIS_DATA_TOOLBOX_WIDGET_HPP_
#define PLI_VIS_DATA_TOOLBOX_WIDGET_HPP_

#include <QWidget>

#include <hdf5/hdf5_io.hpp>

#include <ui_data_toolbox.h>

namespace pli
{
class data_toolbox_widget : public QWidget, public Ui_data_toolbox
{
public:
  data_toolbox_widget(QWidget* parent = nullptr);
};
}

#endif
