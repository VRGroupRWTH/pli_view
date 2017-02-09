#include /* implements */ <ui/viewer.hpp>

#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>

#include <graphics/fdm_factory.hpp>

namespace pli
{
viewer::viewer(QWidget* parent) : QVTKWidget(parent)
{
  renderer_ = vtkSmartPointer<vtkRenderer>::New();

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
  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  auto actor  = vtkSmartPointer<vtkActor>         ::New();

  boost::multi_array<float, 4> coefficients(boost::extents[1][1][1][1]);
  coefficients[0][0][0][0] = 1;
  auto samples = fdm_factory::sample_coefficients(coefficients, {100, 100});
  mapper->SetInputData(fdm_factory::create(samples, {100, 100}));
  actor ->SetMapper   (mapper);

  orientation_marker_ = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
  orientation_marker_->SetOrientationMarker(actor);
  orientation_marker_->SetInteractor       (GetInteractor());
  orientation_marker_->SetViewport         (0.0, 0.0, 0.2, 0.2);
  orientation_marker_->SetEnabled          (true);
  orientation_marker_->InteractiveOff      ();
}
}
