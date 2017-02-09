#ifndef PLI_VIS_VIEWER_HPP_
#define PLI_VIS_VIEWER_HPP_

#include <QVTKWidget.h>
#include <vtkRenderer.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkSmartPointer.h>

#include <attributes/loggable.hpp>

namespace pli
{
class viewer : public QVTKWidget, public loggable<viewer>
{
public:
  viewer(QWidget* parent = nullptr);

  vtkRenderer* renderer() const;

private:
  void create_orientation_marker();

  vtkSmartPointer<vtkRenderer>                renderer_;
  vtkSmartPointer<vtkOrientationMarkerWidget> orientation_marker_;
};
}

#endif
