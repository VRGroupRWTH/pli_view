#include <pli_vis/visualization/algorithms/ospray_streamline_exporter.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <algorithm>

#include <pli_vis/third_party/stb/stb_image_write.h>

namespace pli
{
void ospray_streamline_exporter::set_data(
  const std::vector<float3>& vertices, 
  const std::vector<float3>& tangents)
{
  vertices_ = vertices;
  tangents_ = tangents;
}

void ospray_streamline_exporter::set_camera(
  const float3& position, 
  const float3& forward , 
  const float3& up      )
{
  camera_position_ = position;
  camera_forward_  = forward ;
  camera_up_       = up      ;
}
void ospray_streamline_exporter::set_image_size(
  const uint2& size)
{
  image_size_ = size;
}

void ospray_streamline_exporter::save(
  const std::string& filepath)
{
  std::vector<float4> vertices (vertices_.size());
  std::vector<float4> colors   (tangents_.size());
  std::vector<int>    indices  (vertices_.size() / 2);
  std::transform(vertices_.begin(), vertices_.end(), vertices.begin(), [ ] (const float3& value)
  {
    return float4 {value.x, value.y, value.z, 0.0F};
  });
  std::transform(tangents_.begin(), tangents_.end(), colors  .begin(), [ ] (const float3& value)
  {
    return float4 {abs(value.x), abs(value.z), abs(value.y), 1.0F};
  });
  std::generate (indices  .begin(), indices  .end(), [n = -2] () mutable { n += 2; return n; });

  // Setup renderer.
  ospray::cpp::Renderer renderer("scivis");
  renderer.set   ("oneSidedLighting", true);
  renderer.set   ("shadowsEnabled"  , true);
  renderer.set   ("aoSamples"       , 8   );
  renderer.set   ("spp"             , 16  );
  renderer.set   ("bgColor"         , 1.0F, 1.0F, 1.0F, 1.0F);
  renderer.commit();
  
  // Setup model.
  ospray::cpp::Geometry streamlines("streamlines");
  ospray::cpp::Data     vertex_data(vertices.size(), OSP_FLOAT3A, vertices.data());
  ospray::cpp::Data     color_data (colors  .size(), OSP_FLOAT4 , colors  .data());
  ospray::cpp::Data     index_data (indices .size(), OSP_INT    , indices .data());
  vertex_data.commit();
  color_data .commit();
  index_data .commit();
  streamlines.set("radius"      , 0.1F       );
  streamlines.set("vertex"      , vertex_data);
  streamlines.set("vertex.color", color_data );
  streamlines.set("index"       , index_data );
  
  auto material = renderer.newMaterial("OBJMaterial");
  material   .set        ("Ks", 0.5F, 0.5F, 0.5F);
  material   .set        ("Ns", 2.0F);
  material   .commit     ();
  streamlines.setMaterial(material);
  streamlines.commit     ();
  
  ospray::cpp::Model model;
  model   .addGeometry(streamlines);
  model   .commit     ();
  renderer.set        ("model", model);
  renderer.commit     ();
  
  // Setup camera.
  const ospcommon::vec3f position { camera_position_.x,  camera_position_.y,  camera_position_.z};
  const ospcommon::vec3f forward  {-camera_forward_ .x, -camera_forward_ .y, -camera_forward_ .z};
  const ospcommon::vec3f up       { camera_up_      .x,  camera_up_      .y,  camera_up_      .z};
  const ospcommon::vec2f start    {0.0F, 1.0F};
  const ospcommon::vec2f end      {1.0F, 0.0F};
  ospray::cpp::Camera camera("perspective");
  camera  .set   ("aspect"    , image_size_.x / static_cast<float>(image_size_.y));
  camera  .set   ("fovy"      , 68.0F   );
  camera  .set   ("pos"       , position);
  camera  .set   ("dir"       , forward );
  camera  .set   ("up"        , up      );
  camera  .set   ("imageStart", start   );
  camera  .set   ("imageEnd"  , end     );
  camera  .commit();
  renderer.set   ("camera"    , camera  );
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
  ospray::cpp::FrameBuffer framebuffer(ospcommon::vec2i(image_size_.x, image_size_.y), OSP_FB_SRGBA, OSP_FB_COLOR | OSP_FB_ACCUM);
  framebuffer.clear(OSP_FB_COLOR | OSP_FB_ACCUM);

  // Render.
  for(auto i = 0; i < 2; ++i)
    renderer.renderFrame(framebuffer, OSP_FB_COLOR | OSP_FB_ACCUM);
  
  // Save.
  const auto bytes     = static_cast<uint32_t*>(framebuffer.map(OSP_FB_COLOR));
  const auto extension = filepath.substr(filepath.find_last_of("."));
  if      (extension == ".bmp")
    stbi_write_bmp(filepath.c_str(), image_size_.x, image_size_.y, 4, bytes);
  else if (extension == ".jpg")
    stbi_write_jpg(filepath.c_str(), image_size_.x, image_size_.y, 4, bytes, image_size_.x * sizeof(uint32_t));
  else if (extension == ".png")
    stbi_write_png(filepath.c_str(), image_size_.x, image_size_.y, 4, bytes, image_size_.x * sizeof(uint32_t));
  else if (extension == ".tga")
    stbi_write_tga(filepath.c_str(), image_size_.x, image_size_.y, 4, bytes);
  framebuffer.unmap(bytes);
}
}