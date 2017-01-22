#ifndef PLI_VIS_FDM_PLUGIN_HPP_
#define PLI_VIS_FDM_PLUGIN_HPP_

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_fdm_toolbox.h>

#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>

namespace pli
{
class fdm_plugin : public plugin, public loggable<fdm_plugin>, public Ui_fdm_toolbox
{
public:
  fdm_plugin(QWidget* parent = nullptr);
  
  void start() override;

private:
  void update_viewer() const;
  void calculate    () const;

  vtkSmartPointer<vtkPolyData>       poly_data_;
  vtkSmartPointer<vtkPolyDataMapper> mapper_;
  vtkSmartPointer<vtkActor>          actor_;
};
}

#endif
