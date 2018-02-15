#include <pli_vis/visualization/algorithms/ospray_streamline_renderer.hpp>

#include <numeric>
#include <vector>

#include <pli_vis/third_party/glew/GL/glew.h>
#include <pli_vis/visualization/primitives/camera.hpp>

#include <glm/glm.hpp>
#include <ospray/ospray.h>

namespace pli
{
void ospray_streamline_renderer::initialize()
{
  glm::ivec4 viewport;
  glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));
  glm::ivec2 size(viewport[2], viewport[3]);
  
  camera_ = ospNewCamera("perspective");
  ospSet1f (camera_, "aspect", size[0] / static_cast<float>(size[1]));
  ospCommit(camera_);
  
  renderer_    = ospNewRenderer("scivis");
  streamlines_ = ospNewGeometry("streamlines");
  
  auto model = ospNewModel();
  ospAddGeometry(model, streamlines_);
  set_data({{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}}, {{1.0F, 1.0F, 1.0F}, {1.0F, 1.0F, 1.0F}});
  ospCommit(model);
  
  auto ambient_light = ospNewLight(renderer_, "ambient");
  ospSet1f (ambient_light, "intensity", 0.2);
  ospCommit(ambient_light);
  
  auto distant_light = ospNewLight(renderer_, "distant");
  ospSetVec3f(distant_light, "direction"      , osp::vec3f {1.0F, 1.0F, -0.5F});
  ospSetVec3f(distant_light, "color"          , osp::vec3f {1.0F, 1.0F,  0.8F});
  ospSet1f   (distant_light, "intensity"      , 0.8);
  ospSet1f   (distant_light, "angularDiameter", 1);
  ospCommit  (distant_light);
  
  auto lights_list = std::vector<OSPLight> {ambient_light, distant_light};
  auto lights      = ospNewData(lights_list.size(), OSP_LIGHT, lights_list.data(), 0);
  ospCommit(lights);
  
  ospSetObject(renderer_, "model"         , model  );
  ospSetObject(renderer_, "camera"        , camera_);
  ospSetObject(renderer_, "lights"        , lights );
  ospSet1i    (renderer_, "shadowsEnabled", 1);
  ospSet1i    (renderer_, "aoSamples"     , 8);
  ospSet1i    (renderer_, "spp"           , 16);
  ospSetVec4f (renderer_, "bgColor"       , osp::vec4f {0.0F, 0.0F, 0.0F, 1.0F});
  ospCommit   (renderer_);
}
void ospray_streamline_renderer::render    (const camera* camera)
{
  glm::ivec4 viewport;
  glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));
  glm::ivec2 size(viewport[2], viewport[3]);
  
  auto camera_position = camera->translation();
  auto camera_forward  = camera->forward    ();
  auto camera_up       = camera->up         ();
  ospSet1f   (camera_, "aspect", size[0] / static_cast<float>(size[1]));
  ospSetVec3f(camera_, "pos"   , reinterpret_cast<osp::vec3f&>(camera_position));
  ospSetVec3f(camera_, "dir"   , reinterpret_cast<osp::vec3f&>(camera_forward ));
  ospSetVec3f(camera_, "up"    , reinterpret_cast<osp::vec3f&>(camera_up      ));
  ospCommit  (camera_);
  
  auto framebuffer = ospNewFrameBuffer(reinterpret_cast<osp::vec2i&>(size), OSP_FB_SRGBA, OSP_FB_COLOR);
  ospFrameBufferClear(framebuffer,            OSP_FB_COLOR);
  ospRenderFrame     (framebuffer, renderer_, OSP_FB_COLOR);
  auto bytes = static_cast<const uint32_t*>(ospMapFrameBuffer(framebuffer, OSP_FB_COLOR));
  // TODO: Render to opengl backbuffer directly.
  ospUnmapFrameBuffer(bytes, framebuffer);
}
  
void ospray_streamline_renderer::set_data(
  const std::vector<float3>& points    , 
  const std::vector<float3>& directions)
{
  std::vector<int> indices(points.size());
  std::iota(indices.begin(), indices.end(), 0);
  
  auto vertex_data = ospNewData(points    .size(), OSP_FLOAT3, points    .data(), 0);
  auto color_data  = ospNewData(directions.size(), OSP_FLOAT3, directions.data(), 0);
  auto index_data  = ospNewData(indices   .size(), OSP_UINT  , indices   .data(), 0);
  ospCommit(vertex_data);
  ospCommit(color_data );
  ospCommit(index_data );
  
  ospSet1f  (streamlines_, "radius"      , 2          );
  ospSetData(streamlines_, "vertex"      , vertex_data);
  ospSetData(streamlines_, "vertex.color", color_data );
  ospSetData(streamlines_, "index"       , index_data );
  
  auto material = ospNewMaterial(renderer_, "OBJMaterial");
  ospSetVec3f   (material, "Ks", osp::vec3f{0.5, 0.5, 0.5});
  ospSet1f      (material, "Ns", 2.f);
  ospCommit     (material);
  ospSetMaterial(streamlines_, material);
  
  ospCommit(streamlines_);
}
}