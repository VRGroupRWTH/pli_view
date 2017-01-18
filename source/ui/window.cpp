#include /* implements */ <ui/window.hpp>

#include <QFileDialog>
#include <QVTKWidget.h>

#include <vtkActor.h>
#include <vtkHedgeHog.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>

#include <adapters/qt/line_edit_utility.hpp>
#include <adapters/spdlog/qt_text_browser_sink.hpp>
#include <adapters/vtk/fom_factory.hpp>

namespace pli
{
window:: window()
{
  setupUi      (this);
  showMaximized();
  set_sink     (std::make_shared<qt_text_browser_sink>(console));
  bind_actions ();
}

void window::bind_actions ()
{      
  connect(action_file_open   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Opening file dialog."));
    auto filename = QFileDialog::getOpenFileName(this, tr("Select PLI file."), "C:/", tr("HDF5 Files (*.h5)"));
    logger_->info("Closing file dialog. Selection: {}.", filename.toStdString());
    if (filename.isEmpty())
      return;

    io_.reset(new hdf5_io<float>(
      filename.toStdString(),
      line_edit_utility::get_text(toolbox_data->line_edit_voxel_size   ),
      line_edit_utility::get_text(toolbox_data->line_edit_mask         ),
      line_edit_utility::get_text(toolbox_data->line_edit_transmittance),
      line_edit_utility::get_text(toolbox_data->line_edit_retardation  ),
      line_edit_utility::get_text(toolbox_data->line_edit_direction    ),
      line_edit_utility::get_text(toolbox_data->line_edit_inclination  ),
      line_edit_utility::get_text(toolbox_data->line_edit_distribution )
    ));
    offset_[0] = line_edit_utility::get_text<int>  (toolbox_data->line_edit_offset_x);
    offset_[1] = line_edit_utility::get_text<int>  (toolbox_data->line_edit_offset_y);
    offset_[2] = line_edit_utility::get_text<int>  (toolbox_data->line_edit_offset_z);
    size_  [0] = line_edit_utility::get_text<int>  (toolbox_data->line_edit_size_x  );
    size_  [1] = line_edit_utility::get_text<int>  (toolbox_data->line_edit_size_y  );
    size_  [2] = line_edit_utility::get_text<int>  (toolbox_data->line_edit_size_z  );

    fom_scale_ = line_edit_utility::get_text<float>(toolbox_fom->line_edit_scale    );
    fom_show_  = toolbox_fom->checkbox_show->isChecked();
    
    fom_scale_ = line_edit_utility::get_text<float>(toolbox_fom->line_edit_scale    );
    fom_show_  = toolbox_fom->checkbox_show->isChecked();

    update_viewer();
  });
  connect(action_file_exit   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Closing window."));
    close();
  });
  connect(action_edit_undo   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Undoing last action."));
    // TODO.
  });
  connect(action_edit_redo   , &QAction::triggered, [&] 
  {
    logger_->info(std::string("Redoing last action."));
    // TODO.
  });
  connect(action_help_version, &QAction::triggered, [&] 
  {
    logger_->info(std::string("Version 1.0."));
  });
  
  connect(toolbox_data->line_edit_voxel_size     , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(toolbox_data->line_edit_voxel_size);
    io_->set_attribute_path_voxel_size(text);
    logger_->info("Voxel size attribute path is set to " + text);
    update_viewer();
  });
  connect(toolbox_data->line_edit_mask           , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(toolbox_data->line_edit_mask);
    io_->set_dataset_path_mask(text);
    logger_->info("Mask dataset path is set to " + text);
    update_viewer();
  });
  connect(toolbox_data->line_edit_transmittance  , &QLineEdit::editingFinished, [&] 
  {
    auto text = line_edit_utility::get_text(toolbox_data->line_edit_transmittance);
    io_->set_dataset_path_transmittance(text);
    logger_->info("Transmittance dataset path is set to " + text);
    update_viewer();
  });
  connect(toolbox_data->line_edit_retardation    , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(toolbox_data->line_edit_retardation);
    io_->set_dataset_path_retardation(text);
    logger_->info("Retardation dataset path is set to " + text);
    update_viewer();
  });
  connect(toolbox_data->line_edit_direction      , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(toolbox_data->line_edit_direction);
    io_->set_dataset_path_fiber_direction(text);
    logger_->info("Fiber direction dataset path is set to " + text);
    update_viewer();
  });
  connect(toolbox_data->line_edit_inclination    , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(toolbox_data->line_edit_inclination);
    io_->set_dataset_path_fiber_inclination(text);
    logger_->info("Fiber inclination dataset path is set to " + text);
    update_viewer();
  });
  connect(toolbox_data->line_edit_distribution   , &QLineEdit::editingFinished, [&]
  {
    auto text = line_edit_utility::get_text(toolbox_data->line_edit_distribution);
    io_->set_dataset_path_fiber_distribution(text);
    logger_->info("Fiber distribution dataset path is set to " + text);
    update_viewer();
  });
  connect(toolbox_data->line_edit_offset_x       , &QLineEdit::editingFinished, [&]
  {
    offset_[0] = line_edit_utility::get_text<std::size_t>(toolbox_data->line_edit_offset_x);
    update_viewer();
  });
  connect(toolbox_data->line_edit_offset_y       , &QLineEdit::editingFinished, [&]
  {
    offset_[1] = line_edit_utility::get_text<std::size_t>(toolbox_data->line_edit_offset_y);
    update_viewer();
  });
  connect(toolbox_data->line_edit_offset_z       , &QLineEdit::editingFinished, [&]
  {
    offset_[2] = line_edit_utility::get_text<std::size_t>(toolbox_data->line_edit_offset_z);
    update_viewer();
  });
  connect(toolbox_data->line_edit_size_x         , &QLineEdit::editingFinished, [&]
  {
    size_  [0] = line_edit_utility::get_text<std::size_t>(toolbox_data->line_edit_size_x);
    update_viewer();
  });
  connect(toolbox_data->line_edit_size_y         , &QLineEdit::editingFinished, [&]
  {
    size_  [1] = line_edit_utility::get_text<std::size_t>(toolbox_data->line_edit_size_y);
    update_viewer();
  });
  connect(toolbox_data->line_edit_size_z         , &QLineEdit::editingFinished, [&]
  {
    size_  [2] = line_edit_utility::get_text<std::size_t>(toolbox_data->line_edit_size_z);
    update_viewer();
  });

  connect(toolbox_fom->checkbox_show             , &QCheckBox::stateChanged   , [&](int state)
  {
    fom_show_ = state;
    update_viewer();
  });
  connect(toolbox_fom->line_edit_scale           , &QLineEdit::editingFinished, [&] 
  {
    fom_scale_ = line_edit_utility::get_text<float>(toolbox_fom->line_edit_scale);
    update_viewer();
  });

  connect(toolbox_fdm->checkbox_show             , &QCheckBox::stateChanged   , [&](int state)
  {
    fdm_show_ = state;
    update_viewer();
  });
  connect(toolbox_fdm->line_edit_block_size_x    , &QLineEdit::editingFinished, [&] 
  {
    fdm_block_size_[0]     = line_edit_utility::get_text<std::size_t>(toolbox_fdm->line_edit_block_size_x);
    update_viewer();
  });
  connect(toolbox_fdm->line_edit_block_size_y    , &QLineEdit::editingFinished, [&] 
  {
    fdm_block_size_[1]     = line_edit_utility::get_text<std::size_t>(toolbox_fdm->line_edit_block_size_y);
    update_viewer();
  });
  connect(toolbox_fdm->line_edit_block_size_z    , &QLineEdit::editingFinished, [&] 
  {
    fdm_block_size_[2]     = line_edit_utility::get_text<std::size_t>(toolbox_fdm->line_edit_block_size_z);
    update_viewer();
  });
  connect(toolbox_fdm->line_edit_histogram_bins_x, &QLineEdit::editingFinished, [&] 
  {
    fdm_histogram_bins_[0] = line_edit_utility::get_text<std::size_t>(toolbox_fdm->line_edit_histogram_bins_x);
    update_viewer();
  });
  connect(toolbox_fdm->line_edit_histogram_bins_y, &QLineEdit::editingFinished, [&]
  {
    fdm_histogram_bins_[1] = line_edit_utility::get_text<std::size_t>(toolbox_fdm->line_edit_histogram_bins_y);
    update_viewer();
  });
  connect(toolbox_fdm->line_edit_max_order       , &QLineEdit::editingFinished, [&]
  {
    fdm_max_order_         = line_edit_utility::get_text<std::size_t>(toolbox_fdm->line_edit_max_order);
    update_viewer();
  });
  connect(toolbox_fdm->line_edit_samples_x       , &QLineEdit::editingFinished, [&]
  {
    fdm_samples_[0]        = line_edit_utility::get_text<std::size_t>(toolbox_fdm->line_edit_samples_x);
    update_viewer();
  });
  connect(toolbox_fdm->line_edit_samples_y       , &QLineEdit::editingFinished, [&]
  {
    fdm_samples_[1]        = line_edit_utility::get_text<std::size_t>(toolbox_fdm->line_edit_samples_y);
    update_viewer();
  });
}
void window::update_viewer() const
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
    if (fom_show_)
    {
      hedgehog  ->SetInputData      (fom_factory::create(fiber_direction_map, fiber_inclination_map, voxel_size));
      hedgehog  ->SetScaleFactor    (fom_scale_); 
      mapper    ->SetInputConnection(hedgehog->GetOutputPort());
      actor     ->SetMapper         (mapper);
    }
    renderer->AddActor       (actor);
    renderer->ResetCamera    ();
    viewer  ->GetRenderWindow()->AddRenderer(renderer);
    viewer  ->GetRenderWindow()->Render     ();
  }
  catch (std::exception& ex)
  {
    logger_->error(std::string(ex.what()));
  }
}
}