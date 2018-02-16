#include <pli_vis/visualization/algorithms/ospray_streamline_exporter.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <algorithm>
#include <numeric>
#include <random>

#include <pli_vis/opengl/opengl.hpp>
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

void ospray_streamline_exporter::save(
  const std::string& filepath)
{
  {
    camera_position_ = float3{-248.0F, -62.0F, -60.0F};
    camera_forward_  = float3{125.0F - camera_position_.x, 125.0F - camera_position_.y, 125.0F - camera_position_.z};
    camera_up_       = float3{0.0F, 1.0F, 0.0F};
  }

  std::vector<float4> vertices (vertices_.size());
  std::vector<float4> colors   (tangents_.size());
  std::vector<int>    indices  (vertices_.size() / 2);
  std::transform(vertices_.begin(), vertices_.end(), vertices.begin(), [ ] (const float3& value)
  {
    return float4 {value.x, value.y, value.z, 0.0F};
  });
  std::transform(tangents_.begin(), tangents_.end(), colors  .begin(), [ ] (const float3& value)
  {
    return float4 {abs(value.x), abs(value.y), abs(value.z), 1.0F};
  });
  std::generate (indices  .begin(), indices  .end(), [n = -2] () mutable { n += 2; return n; });

  {
    vertices.clear();
    colors  .clear();
    indices .clear();
  
	  const ospcommon::vec3i volume_dims(256, 256, 256);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> pos_x(-0.05 * volume_dims.x, 1.05 * volume_dims.x);
    std::uniform_real_distribution<float> pos_y(-0.05 * volume_dims.y, 1.05 * volume_dims.y);
    std::uniform_real_distribution<float> pos_z(-0.05 * volume_dims.z, 1.05 * volume_dims.z);
    std::uniform_real_distribution<float> cols (0, 1.0);
    for (size_t i = 0; i < 200; ++i) 
    {
      vertices.push_back(float4{pos_x(rng), pos_y(rng), pos_z(rng), 1.0F});
      colors  .push_back(float4{cols(rng), cols(rng), cols(rng), 1.0});
    }
    const std::uniform_int_distribution<int> streamline_length(1, 5);
    auto next_streamline = 0;
    for (size_t i = 0; i < 20; ++i) 
    {
      const auto length = streamline_length(rng);
      for (auto j = 0; j < length && next_streamline < vertices.size(); ++j, ++next_streamline)
        indices.push_back(next_streamline);
      ++next_streamline;
    }
  }

  int4  viewport; glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));
  uint2 size {viewport.z, viewport.w};
  
  // Setup renderer.
  ospray::cpp::Renderer renderer("scivis");
  renderer.set   ("shadowsEnabled", 1 );
  renderer.set   ("aoSamples"     , 8 );
  renderer.set   ("spp"           , 16);
  renderer.set   ("bgColor"       , 0.02F, 0.02F, 0.02F, 1.0F);
  renderer.commit();
  
  // Setup model.
  ospray::cpp::Geometry streamlines("streamlines");
  ospray::cpp::Data     vertex_data(vertices.size(), OSP_FLOAT3A, vertices.data());
  ospray::cpp::Data     color_data (colors  .size(), OSP_FLOAT4 , colors  .data());
  ospray::cpp::Data     index_data (indices .size(), OSP_INT    , indices .data());
  vertex_data.commit();
  color_data .commit();
  index_data .commit();
  streamlines.set("radius"      , 1.0F       );
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
  const ospcommon::vec3f position {camera_position_.x, camera_position_.y, camera_position_.z};
  const ospcommon::vec3f forward  {camera_forward_ .x, camera_forward_ .y, camera_forward_ .z};
  const ospcommon::vec3f up       {camera_up_      .x, camera_up_      .y, camera_up_      .z};
  ospray::cpp::Camera camera("perspective");
  camera  .set   ("aspect", size.x / static_cast<float>(size.y));
  camera  .set   ("pos"   , position);
  camera  .set   ("dir"   , forward );
  camera  .set   ("up"    , up      );
  camera  .commit();
  renderer.set   ("camera", camera  );
  renderer.commit();
  
  // Setup lights.
  auto ambient_light = renderer.newLight("ambient");
  ambient_light.set   ("intensity", 0.2F);
  ambient_light.commit();
  auto ambient_handle = ambient_light.handle();
  
  //auto distant_light = renderer.newLight("distant");
  //distant_light.set   ("direction"      , 1.0F, 1.0F, -0.5F);
  //distant_light.set   ("color"          , 1.0F, 1.0F,  0.8F);
  //distant_light.set   ("intensity"      , 0.8F);
  //distant_light.set   ("angularDiameter", 1.0F);
  //distant_light.commit();
  //auto distant_handle = distant_light.handle();

  std::vector<OSPLight> lights_list = {ambient_handle};
  ospray::cpp::Data lights(lights_list.size(), OSP_LIGHT, lights_list.data());
  lights  .commit();
  renderer.set   ("lights", lights);
  renderer.commit();
  
  // Setup framebuffer.
  ospray::cpp::FrameBuffer framebuffer(ospcommon::vec2i(size.x, size.y), OSP_FB_SRGBA, OSP_FB_COLOR);
  framebuffer.clear(OSP_FB_COLOR);

  // Render.
  renderer.renderFrame(framebuffer, OSP_FB_COLOR);
  
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