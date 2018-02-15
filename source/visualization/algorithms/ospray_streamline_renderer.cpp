#include <pli_vis/visualization/algorithms/ospray_streamline_renderer.hpp>

#include <numeric>
#include <vector>

#include <pli_vis/third_party/glew/GL/glew.h>
#include <pli_vis/visualization/primitives/camera.hpp>

#include <glm/glm.hpp>
#include <ospray/ospray_cpp.h>
#include <ospray/ospcommon/vec.h>

namespace pli
{
void ospray_streamline_renderer::initialize()
{
  glm::ivec4 viewport;
  glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));
  glm::ivec2 size(viewport[2], viewport[3]);

  texture_. reset    (new gl::texture_2d);
  texture_->bind     ();
  texture_->set_image(GL_RGBA, size[0], size[1], GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  texture_->unbind   ();
  
  // Setup renderer.
  renderer_ = std::make_unique<ospray::cpp::Renderer>("scivis");
  renderer_->set   ("shadowsEnabled", 1 );
  renderer_->set   ("aoSamples"     , 8 );
  renderer_->set   ("spp"           , 16);
  renderer_->set   ("bgColor"       , 0.0F, 0.0F, 0.0F, 1.0F);
  renderer_->commit();
  
  // Setup model.
  streamlines_ = std::make_unique<ospray::cpp::Geometry>("streamlines");
  set_data({{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}}, {{1.0F, 1.0F, 1.0F}, {1.0F, 1.0F, 1.0F}});

  model_ = std::make_unique<ospray::cpp::Model>();
  model_   ->addGeometry(*streamlines_.get());
  model_   ->commit     ();
  renderer_->set        ("model", *model_.get());
  renderer_->commit     ();

  // Setup camera.
  camera_ = std::make_unique<ospray::cpp::Camera>("perspective");
  camera_  ->set   ("aspect", size[0] / static_cast<float>(size[1]));
  camera_  ->commit();
  renderer_->set   ("camera", *camera_.get());
  renderer_->commit();

  // Setup lights.
  auto ambient_light = renderer_->newLight("ambient");
  ambient_light.set   ("intensity"      , 0.2F);
  ambient_light.commit();
  
  auto distant_light = renderer_->newLight("distant");
  distant_light.set   ("direction"      , 1.0F, 1.0F, -0.5F);
  distant_light.set   ("color"          , 1.0F, 1.0F,  0.8F);
  distant_light.set   ("intensity"      , 0.8F);
  distant_light.set   ("angularDiameter", 1.0F);
  distant_light.commit();

  auto lights = std::vector<ospray::cpp::Light>{ambient_light, distant_light};
  lights_ = std::make_unique<ospray::cpp::Data>(lights.size(), OSP_LIGHT, lights.data(), 0);
  lights_  ->commit();
  renderer_->set   ("lights", lights_.get());
  renderer_->commit();

  // Setup framebuffer.
  framebuffer_ = std::make_unique<ospray::cpp::FrameBuffer>(ospcommon::vec2i(size[0], size[1]), OSP_FB_SRGBA, OSP_FB_COLOR);
}
void ospray_streamline_renderer::render    (const camera* camera)
{
  glm::ivec4 viewport;
  glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));
  glm::ivec2 size(viewport[2], viewport[3]);
  
  auto camera_position = camera->translation();
  auto camera_forward  = camera->forward    ();
  auto camera_up       = camera->up         ();
  camera_->set   ("aspect", size[0] / static_cast<float>(size[1]));
  camera_->set   ("pos"   , camera_position[0], camera_position[1], camera_position[2]);
  camera_->set   ("dir"   , camera_forward [0], camera_forward [1], camera_forward [2]);
  camera_->set   ("up"    , camera_up      [0], camera_up      [1], camera_up      [2]);
  camera_->commit();

  framebuffer_->clear      (OSP_FB_COLOR);
  renderer_   ->renderFrame(*framebuffer_.get(), 0);

  const auto bytes = static_cast<uint32_t*>(framebuffer_->map(OSP_FB_COLOR));
  texture_    ->bind     ();
  texture_    ->set_image(GL_RGBA, size[0], size[1], GL_RGBA, GL_UNSIGNED_BYTE, bytes);
  texture_    ->unbind   ();
  framebuffer_->unmap    (bytes);
}
  
void ospray_streamline_renderer::set_data(
  const std::vector<float3>& points    , 
  const std::vector<float3>& directions)
{
  std::vector<unsigned> indices(points.size());
  std::iota(indices.begin(), indices.end(), 0);
  
  auto vertex_data = ospray::cpp::Data(points    .size(), OSP_FLOAT3, points    .data()); 
  auto color_data  = ospray::cpp::Data(directions.size(), OSP_FLOAT3, directions.data()); 
  auto index_data  = ospray::cpp::Data(indices   .size(), OSP_UINT  , indices   .data()); 
  vertex_data.commit();
  color_data .commit();
  index_data .commit();

  streamlines_->set("radius"      , 2.0F       );
  streamlines_->set("vertex"      , vertex_data);
  streamlines_->set("vertex.color", color_data );
  streamlines_->set("index"       , index_data );
  
  auto material = renderer_->newMaterial("OBJMaterial");
  material.set   ("Ks", 0.5F, 0.5F, 0.5F);
  material.set   ("Ns", 2.0F);
  material.commit();
  streamlines_->setMaterial(material);
  streamlines_->commit     ();
}
}