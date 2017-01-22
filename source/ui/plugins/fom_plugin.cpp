#include /* implements */ <ui/plugins/fom_plugin.hpp>

#include <limits>

#include <vtkSmartPointer.h>

#include <ui/window.hpp>
#include <utility/qt/line_edit_utility.hpp>
#include <utility/spdlog/qt_text_browser_sink.hpp>
#include <utility/vtk/fom_factory.hpp>

namespace pli
{
fom_plugin::fom_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  line_edit_offset_x->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_y->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_z->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_x  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_y  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_z  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_scale   ->setValidator(new QDoubleValidator(0, std::numeric_limits<double>::max(), 10, this));
  
  connect(line_edit_offset_x, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection X offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_x));
    update_viewer();
  });
  connect(line_edit_offset_y, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Y offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_y));
    update_viewer();
  });
  connect(line_edit_offset_z, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Z offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_z));
    update_viewer();
  });
  connect(line_edit_size_x  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection X size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_x));
    update_viewer();
  });
  connect(line_edit_size_y  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Y size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_y));
    update_viewer();
  });
  connect(line_edit_size_z  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Selection Z size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_z));
    update_viewer();
  });
  connect(checkbox_show     , &QCheckBox::stateChanged   , [&] (int state)
  {
    logger_->info(std::string("Show set to ") + (state ? "true" : "false"));
    update_viewer();
  });
  connect(line_edit_scale   , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Vector scale set to " + line_edit_scale->text().toStdString());
    update_viewer();
  });

  hedgehog_ = vtkSmartPointer<vtkHedgeHog>      ::New();
  mapper_   = vtkSmartPointer<vtkPolyDataMapper>::New();
  actor_    = vtkSmartPointer<vtkActor>         ::New();
}

void fom_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    logger_->info(std::string("Updating viewer."));
    update_viewer();
  });

  owner_->viewer->renderer()->AddActor(actor_);
  owner_->viewer->renderer()->ResetCamera();
}

void fom_plugin::update_viewer() const
{
  try
  {
    auto data_plugin = owner_->get_plugin<pli::data_plugin>();
    auto io          = data_plugin->io();

    if (io && checkbox_show->isChecked())
    {
      std::array<std::size_t, 3> offset = 
      { line_edit_utility::get_text<std::size_t>(line_edit_offset_x), 
        line_edit_utility::get_text<std::size_t>(line_edit_offset_y),
        line_edit_utility::get_text<std::size_t>(line_edit_offset_z)};
      
      std::array<std::size_t, 3> size = 
      { line_edit_utility::get_text<std::size_t>(line_edit_size_x),
        line_edit_utility::get_text<std::size_t>(line_edit_size_y),
        line_edit_utility::get_text<std::size_t>(line_edit_size_z)};

      auto fiber_direction_map   = io->load_fiber_direction_map  (offset, size);
      auto fiber_inclination_map = io->load_fiber_inclination_map(offset, size);
      auto voxel_size            = io->load_voxel_size           ();
      hedgehog_->SetInputData(fom_factory::create(fiber_direction_map, fiber_inclination_map, voxel_size));
    }
    else
      hedgehog_->SetInputData(vtkSmartPointer<vtkPolyData>::New());
    
    hedgehog_->SetScaleFactor    (line_edit_utility::get_text<float>(line_edit_scale)); 
    mapper_  ->SetInputConnection(hedgehog_->GetOutputPort());
    actor_   ->SetMapper         (mapper_);
    owner_   ->viewer->update();
  }
  catch (std::exception& exception)
  {
    logger_->error(std::string(exception.what()));
  }
}
}
