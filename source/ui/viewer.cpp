#include /* implements */ <ui/viewer.hpp>

#include <vtkRenderWindow.h>

namespace pli
{
viewer::viewer(QWidget* parent) : QVTKWidget(parent)
{
  renderer_ = vtkSmartPointer<vtkRenderer>::New();

  QVTKWidget::GetRenderWindow()->AddRenderer(renderer_);
  QVTKWidget::GetRenderWindow()->Render     ();
}

vtkRenderer* viewer::renderer() const
{
  return renderer_.Get();
}
}
