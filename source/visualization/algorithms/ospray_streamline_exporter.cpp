#include <pli_vis/visualization/algorithms/ospray_streamline_exporter.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <ospray/ospray_cpp.h>

#include <pli_vis/third_party/stb/stb_image_write.h>

namespace pli
{
namespace ospray_streamline_exporter
{
void to_image(
  const float3&                position  , 
  const float3&                forward   , 
  const float3&                up        , 
  const uint2&                 size      ,
  const std::vector<float4>&   vertices  , 
  const std::vector<float4>&   tangents  ,
  const std::vector<unsigned>& gl_indices,
  const std::string&           filepath  )
{
  std::vector<float4> colors (tangents  .size());
  std::vector<int>    indices(gl_indices.size() / 2);
  for (auto i = 0; i < colors .size(); ++i)
    colors [i] = float4{abs(tangents[i].x), abs(tangents[i].z), abs(tangents[i].y), 1.0};
  for (auto i = 0; i < indices.size(); ++i)
    indices[i] = gl_indices[2 * i];

  // Setup renderer.
  ospray::cpp::Renderer renderer("scivis");
  renderer.set   ("oneSidedLighting", false);
  renderer.set   ("shadowsEnabled"  , true );
  renderer.set   ("aoSamples"       , 8    );
  renderer.set   ("spp"             , 16   );
  renderer.set   ("bgColor"         , 1.0F, 1.0F, 1.0F, 1.0F);
  renderer.commit();

  // Setup model.
  ospray::cpp::Geometry streamlines("streamlines");
  ospray::cpp::Data     vertex_data(vertices.size(), OSP_FLOAT3A, vertices.data()); vertex_data.commit();
  ospray::cpp::Data     color_data (colors  .size(), OSP_FLOAT4 , colors  .data()); color_data .commit();
  ospray::cpp::Data     index_data (indices .size(), OSP_INT    , indices .data()); index_data .commit();
  streamlines.set("radius"      , 0.1F       );
  streamlines.set("vertex"      , vertex_data);
  streamlines.set("vertex.color", color_data );
  streamlines.set("index"       , index_data );
  
  auto material = renderer.newMaterial("OBJMaterial");
  material   .set        ("Ks", 0.5F, 0.5F, 0.5F);
  material   .set        ("Ns", 10.0F);
  material   .commit     ();
  streamlines.setMaterial(material);
  streamlines.commit     ();
  
  ospray::cpp::Model model;
  model   .addGeometry(streamlines);
  model   .commit     ();
  renderer.set        ("model", model);
  renderer.commit     ();
  
  // Setup camera.
  const ospcommon::vec3f camera_position { position.x,  position.y,  position.z};
  const ospcommon::vec3f camera_forward  {-forward .x, -forward .y, -forward .z};
  const ospcommon::vec3f camera_up       { up      .x,  up      .y,  up      .z};
  const ospcommon::vec2f start    {0.0F, 1.0F};
  const ospcommon::vec2f end      {1.0F, 0.0F};
  ospray::cpp::Camera camera("perspective");
  camera  .set   ("aspect"    , size.x / static_cast<float>(size.y));
  camera  .set   ("fovy"      , 68.0F          );
  camera  .set   ("pos"       , camera_position);
  camera  .set   ("dir"       , camera_forward );
  camera  .set   ("up"        , camera_up      );
  camera  .set   ("imageStart", start          );
  camera  .set   ("imageEnd"  , end            );
  camera  .commit();                           
  renderer.set   ("camera"    , camera         );
  renderer.commit();
  
  // Setup lights.
  auto ambient_light = renderer.newLight("ambient");
  ambient_light.set   ("intensity", 0.2F);
  ambient_light.commit();
  const auto ambient_handle = ambient_light.handle();
  
  auto distant_light = renderer.newLight("distant");
  distant_light.set   ("direction"      , 1.0F, 1.0F, -0.5F);
  distant_light.set   ("color"          , 1.0F, 1.0F,  0.8F);
  distant_light.set   ("intensity"      , 0.8F);
  distant_light.set   ("angularDiameter", 1.0F);
  distant_light.commit();
  const auto distant_handle = distant_light.handle();
  
  auto distant_light_2 = renderer.newLight("distant");
  distant_light_2.set   ("direction"      , 0.5F, 1.0F, 1.5F);
  distant_light_2.set   ("color"          , 1.0F, 1.0F, 0.8F);
  distant_light_2.set   ("intensity"      , 0.2F);
  distant_light_2.set   ("angularDiameter", 8.0F);
  distant_light_2.commit();
  const auto distant_handle_2 = distant_light_2.handle();

  std::vector<OSPLight> lights_list = {ambient_handle, distant_handle, distant_handle_2};
  ospray::cpp::Data lights(lights_list.size(), OSP_LIGHT, lights_list.data());
  lights  .commit();
  renderer.set   ("lights", lights);
  renderer.commit();
  
  // Setup framebuffer.
  ospray::cpp::FrameBuffer framebuffer(ospcommon::vec2i(size.x, size.y), OSP_FB_SRGBA, OSP_FB_COLOR | OSP_FB_ACCUM);
  framebuffer.clear(OSP_FB_COLOR | OSP_FB_ACCUM);

  // Render.
  for(auto i = 0; i < 2; ++i)
    renderer.renderFrame(framebuffer, OSP_FB_COLOR | OSP_FB_ACCUM);
  
  // Save.
  const auto bytes     = static_cast<uint32_t*>(framebuffer.map(OSP_FB_COLOR));
  const auto extension = filepath.substr(filepath.find_last_of("."));
  if      (extension == ".bmp")
    stbi_write_bmp(filepath.c_str(), size.x, size.y, 4, bytes);
  else if (extension == ".jpg")
    stbi_write_jpg(filepath.c_str(), size.x, size.y, 4, bytes, size.x * sizeof(uint32_t));
  else if (extension == ".png")
    stbi_write_png(filepath.c_str(), size.x, size.y, 4, bytes, size.x * sizeof(uint32_t));
  else if (extension == ".tga")
    stbi_write_tga(filepath.c_str(), size.x, size.y, 4, bytes);
  framebuffer.unmap(bytes);
}
}
}