#ifndef PLI_VIS_FOM_TOOLBOX_WIDGET_HPP_
#define PLI_VIS_FOM_TOOLBOX_WIDGET_HPP_

#include <QWidget>

#include <ui_fom_toolbox.h>

namespace pli
{
class fom_toolbox_widget : public QWidget, public Ui_fom_toolbox
{
public:
  fom_toolbox_widget(QWidget* parent = nullptr);
};
}

#endif
