#ifndef PLI_VIS_FOM_PLUGIN_HPP_
#define PLI_VIS_FOM_PLUGIN_HPP_

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_fom_toolbox.h>

#include <vtkActor.h>
#include <vtkHedgeHog.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>

namespace pli
{
class fom_plugin : public plugin, public loggable<fom_plugin>, public Ui_fom_toolbox
{
public:
  fom_plugin(QWidget* parent = nullptr);

  void start() override;

private:
  void update_viewer() const;

  vtkSmartPointer<vtkHedgeHog>       hedgehog_;
  vtkSmartPointer<vtkPolyDataMapper> mapper_  ;
  vtkSmartPointer<vtkActor>          actor_   ;
};
}

#endif
