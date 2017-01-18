#ifndef PLI_VIS_FDM_TOOLBOX_WIDGET_HPP_
#define PLI_VIS_FDM_TOOLBOX_WIDGET_HPP_

#include <QWidget>

#include <ui_fdm_toolbox.h>

namespace pli
{
class fdm_toolbox_widget : public QWidget, public Ui_fdm_toolbox
{
public:
  fdm_toolbox_widget(QWidget* parent = nullptr);
};
}

#endif
