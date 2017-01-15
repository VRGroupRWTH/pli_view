#include /* implements */ <window.hpp>

#include <QFileDialog>
#include <QVTKWidget.h>

#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

#include <hdf5/hdf5_io.hpp>

#include <sinks/qt_text_browser_sink.hpp>

namespace pli
{
window:: window()
{
  ui_.setupUi  (this);
  set_sink     (std::make_shared<qt_text_browser_sink>(ui_.console));
  bind_actions ();
  showMaximized();
}
window::~window()
{
  
}

void window::bind_actions()
{
  connect(ui_.action_file_open, &QAction::triggered, [&] {
    logger_->info(std::string("Opening file dialog."));
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("HDF5 Files (*.h5)"));
    logger_->info("Closing file dialog. Selection: {}.", filename.toStdString());

    // Read file.
    pli::hdf5_io<float> io(filename.toStdString());
    std::string dataset_path_prefix = "/%Slice%/Microscope/Processed/Registered/";
    io.set_attribute_path_voxel_size     ("DataSpacing");
    io.set_dataset_path_mask             (dataset_path_prefix + "Mask");
    io.set_dataset_path_transmittance    (dataset_path_prefix + "NTransmittance");
    io.set_dataset_path_retardation      (dataset_path_prefix + "Retardation");
    io.set_dataset_path_fiber_direction  (dataset_path_prefix + "Direction");
    io.set_dataset_path_fiber_inclination(dataset_path_prefix + "Inclination");
    std::array<float, 3>         voxel_size;
    boost::multi_array<float, 3> fiber_inclination_map, fiber_direction_map;
    io.load_voxel_size(voxel_size);
    io.load_fiber_inclination_map({{0, 0, 536}}, {{128, 128, 3}}, fiber_inclination_map);
    io.load_fiber_direction_map  ({{0, 0, 536}}, {{128, 128, 3}}, fiber_direction_map  );
    //std::for_each(fiber_direction_map.data(), fiber_direction_map.data() + fiber_direction_map.num_elements(), [](float& elem) { elem++; });
    //io.save_fiber_direction_map  ({{0, 0, 536}}, {{128, 128, 3}}, fiber_direction_map);

    auto source   = vtkSmartPointer<vtkSphereSource>  ::New();
    auto mapper   = vtkSmartPointer<vtkPolyDataMapper>::New();
    auto actor    = vtkSmartPointer<vtkActor>         ::New();
    auto renderer = vtkSmartPointer<vtkRenderer>      ::New();
    mapper    ->SetInputConnection(source->GetOutputPort());
    actor     ->SetMapper         (mapper);
    renderer  ->AddActor          (actor);
    ui_.viewer->GetRenderWindow   ()->AddRenderer(renderer);
    ui_.viewer->show();
  });
  connect(ui_.action_file_exit, &QAction::triggered, [&] {
    logger_->info(std::string("Closing window."));
    close();
  });
  connect(ui_.action_edit_undo, &QAction::triggered, [&] {
    logger_->info(std::string("Undoing last action."));
    // TODO.
  });
  connect(ui_.action_edit_redo, &QAction::triggered, [&] {
    logger_->info(std::string("Redoing last action."));
    // TODO.
  });
  connect(ui_.action_help_version, &QAction::triggered, [&] {
    logger_->info(std::string("Version 1.0."));
  });
}
}