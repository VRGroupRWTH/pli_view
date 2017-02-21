#include /* implements */ <ui/plugins/fom_plugin.hpp>

#include <limits>

#include <all.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>

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
}

void fom_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));

  connect(owner_->get_plugin<data_plugin>(), &data_plugin::on_change, [&]
  {
    logger_->info(std::string("Updating viewer."));
    update_viewer();
  });
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
      auto vector_spacing        = io->load_vector_spacing       ();

      //hedgehog_->SetInputData(fom_factory::create(fiber_direction_map, fiber_inclination_map, vector_spacing));

      gl::array_buffer vertices;
      vertices.bind    ();
      vertices.allocate(sizeof(float3) * fiber_direction_map.num_elements() * 2);
      vertices.unbind  ();

      struct cudaGraphicsResource* vertices_cuda;
      cudaGraphicsGLRegisterBuffer  (&vertices_cuda, vertices.id(), cudaGraphicsMapFlagsWriteDiscard);

      float3* vertices_ptr;
      size_t  num_bytes   ;
      cudaGraphicsMapResources(1, &vertices_cuda, nullptr);
      cudaGraphicsResourceGetMappedPointer((void**)&vertices_ptr, &num_bytes, vertices_cuda);
      
      // TODO: CALL KERNEL ON vertices_ptr.

      cudaGraphicsUnmapResources(1, &vertices_cuda, nullptr);

      // TODO: RENDER.

      cudaGraphicsUnregisterResource(vertices_cuda);
    }
    
    // line_edit_utility::get_text<float>(line_edit_scale)

    owner_->viewer->update();
  }
  catch (std::exception& exception)
  {
    logger_->error(std::string(exception.what()));
  }
}
}
