#include /* implements */ <window.hpp>

#include <boost/lexical_cast.hpp>

#include <QFileDialog>
#include <QVTKWidget.h>

#include <vtkActor.h>
#include <vtkHedgeHog.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>


#include <adapters/fom_poly_data.hpp>
#include <sinks/qt_text_browser_sink.hpp>

std::string get_line_edit_text(QLineEdit* line_edit)
{
  return !line_edit->text           ().isEmpty    () ?
          line_edit->text           ().toStdString() :
          line_edit->placeholderText().toStdString() ;
}

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

void window::bind_actions ()
{
  ui_.line_edit_offset_x->setValidator(new QIntValidator(0, INT_MAX, this));
  ui_.line_edit_offset_y->setValidator(new QIntValidator(0, INT_MAX, this));
  ui_.line_edit_offset_z->setValidator(new QIntValidator(0, INT_MAX, this));
  ui_.line_edit_size_x  ->setValidator(new QIntValidator(0, INT_MAX, this));
  ui_.line_edit_size_y  ->setValidator(new QIntValidator(0, INT_MAX, this));
  ui_.line_edit_size_z  ->setValidator(new QIntValidator(0, INT_MAX, this));

  connect(ui_.action_file_open   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Opening file dialog."));
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("HDF5 Files (*.h5)"));
    logger_->info("Closing file dialog. Selection: {}.", filename.toStdString());

    if (filename.isEmpty())
      return;

    io_.reset(new hdf5_io<float>(filename.toStdString()));
    io_->set_attribute_path_voxel_size      (get_line_edit_text(ui_.line_edit_voxel_size   ));
    io_->set_dataset_path_mask              (get_line_edit_text(ui_.line_edit_mask         ));
    io_->set_dataset_path_transmittance     (get_line_edit_text(ui_.line_edit_transmittance));
    io_->set_dataset_path_retardation       (get_line_edit_text(ui_.line_edit_retardation  ));
    io_->set_dataset_path_fiber_direction   (get_line_edit_text(ui_.line_edit_direction    ));
    io_->set_dataset_path_fiber_inclination (get_line_edit_text(ui_.line_edit_inclination  ));
    io_->set_dataset_path_fiber_distribution(get_line_edit_text(ui_.line_edit_distribution ));
    offset_[0] = boost::lexical_cast<float> (get_line_edit_text(ui_.line_edit_offset_x     ));
    offset_[1] = boost::lexical_cast<float> (get_line_edit_text(ui_.line_edit_offset_y     ));
    offset_[2] = boost::lexical_cast<float> (get_line_edit_text(ui_.line_edit_offset_z     ));
    size_  [0] = boost::lexical_cast<float> (get_line_edit_text(ui_.line_edit_size_x       ));
    size_  [1] = boost::lexical_cast<float> (get_line_edit_text(ui_.line_edit_size_y       ));
    size_  [2] = boost::lexical_cast<float> (get_line_edit_text(ui_.line_edit_size_z       ));
    update_viewer();
  });
  
  connect(ui_.line_edit_voxel_size   , &QLineEdit::editingFinished, [&]
  {
    auto text = get_line_edit_text(ui_.line_edit_voxel_size);
    io_->set_attribute_path_voxel_size(text);
    logger_->info("Voxel size attribute path is set to " + text);
    update_viewer();
  });
  connect(ui_.line_edit_mask         , &QLineEdit::editingFinished, [&]
  {
    auto text = get_line_edit_text(ui_.line_edit_mask);
    io_->set_dataset_path_mask(text);
    logger_->info("Mask dataset path is set to " + text);
    update_viewer();
  });
  connect(ui_.line_edit_transmittance, &QLineEdit::editingFinished, [&] 
  {
    auto text = get_line_edit_text(ui_.line_edit_transmittance);
    io_->set_dataset_path_transmittance(text);
    logger_->info("Transmittance dataset path is set to " + text);
    update_viewer();
  });
  connect(ui_.line_edit_retardation  , &QLineEdit::editingFinished, [&]
  {
    auto text = get_line_edit_text(ui_.line_edit_retardation);
    io_->set_dataset_path_retardation(text);
    logger_->info("Retardation dataset path is set to " + text);
    update_viewer();
  });
  connect(ui_.line_edit_direction    , &QLineEdit::editingFinished, [&]
  {
    auto text = get_line_edit_text(ui_.line_edit_direction);
    io_->set_dataset_path_fiber_direction(text);
    logger_->info("Fiber direction dataset path is set to " + text);
    update_viewer();
  });
  connect(ui_.line_edit_inclination  , &QLineEdit::editingFinished, [&]
  {
    auto text = get_line_edit_text(ui_.line_edit_inclination);
    io_->set_dataset_path_fiber_inclination(text);
    logger_->info("Fiber inclination dataset path is set to " + text);
    update_viewer();
  });
  connect(ui_.line_edit_distribution , &QLineEdit::editingFinished, [&]
  {
    auto text = get_line_edit_text(ui_.line_edit_distribution);
    io_->set_dataset_path_fiber_distribution(text);
    logger_->info("Fiber distribution dataset path is set to " + text);
    update_viewer();
  });

  connect(ui_.line_edit_offset_x     , &QLineEdit::editingFinished, [&]
  {
    offset_[0] = boost::lexical_cast<std::size_t>(get_line_edit_text(ui_.line_edit_offset_x));
    update_viewer();
  });
  connect(ui_.line_edit_offset_y     , &QLineEdit::editingFinished, [&]
  {
    offset_[1] = boost::lexical_cast<std::size_t>(get_line_edit_text(ui_.line_edit_offset_y));
    update_viewer();
  });
  connect(ui_.line_edit_offset_z     , &QLineEdit::editingFinished, [&]
  {
    offset_[2] = boost::lexical_cast<std::size_t>(get_line_edit_text(ui_.line_edit_offset_z));
    update_viewer();
  });
  connect(ui_.line_edit_size_x       , &QLineEdit::editingFinished, [&]
  {
    size_  [0] = boost::lexical_cast<std::size_t>(get_line_edit_text(ui_.line_edit_size_x));
    update_viewer();
  });
  connect(ui_.line_edit_size_y       , &QLineEdit::editingFinished, [&]
  {
    size_  [1] = boost::lexical_cast<std::size_t>(get_line_edit_text(ui_.line_edit_size_y));
    update_viewer();
  });
  connect(ui_.line_edit_size_z       , &QLineEdit::editingFinished, [&]
  {
    size_  [2] = boost::lexical_cast<std::size_t>(get_line_edit_text(ui_.line_edit_size_z));
    update_viewer();
  });

  connect(ui_.action_file_exit   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Closing window."));
    close();
  });
  connect(ui_.action_edit_undo   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Undoing last action."));
    // TODO.
  });
  connect(ui_.action_edit_redo   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Redoing last action."));
    // TODO.
  });
  connect(ui_.action_help_version, &QAction::triggered, [&] 
  {
    logger_->info(std::string("Version 1.0."));
  });
}
void window::update_viewer()
{
  try
  {
    std::array<float, 3>         voxel_size;
    boost::multi_array<float, 3> fiber_direction_map, fiber_inclination_map;
    io_->load_voxel_size           (voxel_size);
    io_->load_fiber_direction_map  (offset_, size_, fiber_direction_map  );
    io_->load_fiber_inclination_map(offset_, size_, fiber_inclination_map);

    auto hedgehog = vtkSmartPointer<vtkHedgeHog>      ::New();
    auto mapper   = vtkSmartPointer<vtkPolyDataMapper>::New();
    auto actor    = vtkSmartPointer<vtkActor>         ::New();
    auto renderer = vtkSmartPointer<vtkRenderer>      ::New();
    hedgehog  ->SetInputData      (fom_poly_data<float>::create(fiber_direction_map, fiber_inclination_map, voxel_size));
    hedgehog  ->SetScaleFactor    (0.001);
    mapper    ->SetInputConnection(hedgehog->GetOutputPort());
    actor     ->SetMapper         (mapper);
    renderer  ->AddActor          (actor);
    ui_.viewer->GetRenderWindow   ()->AddRenderer(renderer);
    ui_.viewer->show              ();
    renderer  ->ResetCamera       ();
    ui_.viewer->GetRenderWindow   ()->Render();
  }
  catch (std::exception& ex)
  {
    logger_->error(std::string(ex.what()));
  }
}

}