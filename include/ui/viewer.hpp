#ifndef PLI_VIS_VIEWER_HPP_
#define PLI_VIS_VIEWER_HPP_

#include <QVTKWidget.h>

#include <vtkRenderer.h>
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
  vtkSmartPointer<vtkRenderer> renderer_;
};
}

#endif
