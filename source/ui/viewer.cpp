#include /* implements */ <ui/viewer.hpp>

#include <vtkCamera.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>

#include <graphics/fdm_factory.hpp>
#include <graphics/sampling.hpp>

namespace pli
{
viewer::viewer(QWidget* parent) : QVTKWidget(parent)
{
  renderer_ = vtkSmartPointer<vtkRenderer>::New();
  //renderer_->GetActiveCamera()->SetParallelProjection(1);

  QVTKWidget::GetRenderWindow()->AddRenderer(renderer_);
  QVTKWidget::GetRenderWindow()->Render     ();

  create_orientation_marker();
}

vtkRenderer* viewer::renderer() const
{
  return renderer_.Get();
}

void viewer::create_orientation_marker()
{
  std::array<size_t, 2> dimensions = {256, 256};
  auto sphere = sample_sphere(dimensions);
  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  auto actor  = vtkSmartPointer<vtkActor>         ::New();
  mapper->SetInputData(fdm_factory::create(sphere, dimensions));
  actor ->SetMapper   (mapper);

  orientation_marker_ = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
  orientation_marker_->SetOrientationMarker(actor);
  orientation_marker_->SetInteractor       (GetInteractor());
  orientation_marker_->SetViewport         (0.0, 0.0, 0.2, 0.2);
  orientation_marker_->SetEnabled          (true);
  orientation_marker_->InteractiveOff      ();
}
}
