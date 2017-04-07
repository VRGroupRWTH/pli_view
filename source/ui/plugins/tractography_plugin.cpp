#include /* implements */ <ui/plugins/tractography_plugin.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

#include <ui/window.hpp>
#include <utility/line_edit_utility.hpp>
#include <utility/qt_text_browser_sink.hpp>
#include <convert.h>

namespace pli
{
tractography_plugin::tractography_plugin(QWidget* parent) : plugin(parent)
{
  setupUi(this);
  
  line_edit_offset_x->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_y->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_offset_z->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_x  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_y  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  line_edit_size_z  ->setValidator(new QIntValidator(0, std::numeric_limits<int>::max(), this));
  
  connect(line_edit_offset_x, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents X offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_x));
  });
  connect(line_edit_offset_y, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents Y offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_y));
  });
  connect(line_edit_offset_z, &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents Z offset is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_offset_z));
  });
  connect(line_edit_size_x  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents X size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_x));
  });
  connect(line_edit_size_y  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents Y size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_y));
  });
  connect(line_edit_size_z  , &QLineEdit::editingFinished, [&]
  {
    logger_->info("Extents Z size is set to {}.", line_edit_utility::get_text<std::size_t>(line_edit_size_z));
  });
  connect(button_trace      , &QPushButton::clicked      , [&]
  {
    trace();
  });
}

void tractography_plugin::start()
{
  set_sink(std::make_shared<qt_text_browser_sink>(owner_->console));
}
void tractography_plugin::trace()
{
  logger_->info(std::string("Starting trace."));

  try
  {
    auto data_plugin = owner_->get_plugin<pli::data_plugin>();
    auto io          = data_plugin->io();

    std::array<std::size_t, 3> offset = 
    { line_edit_utility::get_text<std::size_t>(line_edit_offset_x), 
      line_edit_utility::get_text<std::size_t>(line_edit_offset_y),
      line_edit_utility::get_text<std::size_t>(line_edit_offset_z)};
      
    std::array<std::size_t, 3> size = 
    { line_edit_utility::get_text<std::size_t>(line_edit_size_x),
      line_edit_utility::get_text<std::size_t>(line_edit_size_y),
      line_edit_utility::get_text<std::size_t>(line_edit_size_z)};

    auto fiber_direction_map   = io->load_fiber_direction_dataset  (offset, size);
    auto fiber_inclination_map = io->load_fiber_inclination_dataset(offset, size);
    auto spacing               = io->load_vector_spacing();
    auto shape                 = fiber_direction_map.shape();
    
    // Feed the directions and inclinations into a tangent Cartesian grid.
    tangent::CartesianGrid grid({shape[0], shape[1], shape[2]}, spacing);
    auto grid_ptr = grid.GetVectorPointer(0);
    for(auto x = 0; x < shape[0]; x++)
      for(auto y = 0; y < shape[1]; y++)
        for(auto z = 0; z < shape[2]; z++)
        {
          float longitude = (90.0F + fiber_direction_map  [x][y][z]) * M_PI / 180.0F;
          float latitude  = (90.0F - fiber_inclination_map[x][y][z]) * M_PI / 180.0F;
          auto  cartesian = cush::to_cartesian_coords(float3{1.0F, longitude, latitude});
          grid_ptr[z + shape[2] * (y + shape[1] * x)] = tangent::vector_t{{cartesian.x, cartesian.y, cartesian.z}};
        }

    // Run tangent, record results.
    std::vector<tangent::point_t> seeds;
    for (auto x = offset[0]; x < size[0] + offset[0]; x++)
      for (auto y = offset[1]; y < size[1] + offset[1]; y++)
        for (auto z = offset[2]; z < size[2] + offset[2]; z++)
          seeds.push_back({{float(x), float(y), float(z), 0.0f}});

    linear_tracer tracer;
    tracer.SetData           (&grid);
    tracer.SetIntegrationStep(1.0f );
    auto results = tracer.TraceSeeds(seeds);
     
    // TODO: Visualize recorded results.

         
    owner_->viewer->update();

    logger_->info(std::string("Ending trace."));
  }
  catch (std::exception& exception)
  {
    logger_->error(std::string(exception.what()));
  }
}
}
