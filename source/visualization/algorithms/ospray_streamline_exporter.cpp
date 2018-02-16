#include <pli_vis/visualization/algorithms/ospray_streamline_exporter.hpp>

#include <pli_vis/third_party/stb/stb_image_write.h>

#include <algorithm>

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
  //std::vector<float4> input_vertices    (points    .size());
  //std::transform(points.begin(), points.end(), input_vertices.begin(), [ ] (const float3& value)
  //{
  //  return float4 {value.x, value.y, value.z, 0.0F};
  //});
  //std::vector<float4> input_colors(directions.size());
  //std::transform(directions.begin(), directions.end(), input_colors.begin(), [ ] (const float3& value)
  //{
  //  return float4 {abs(value.x), abs(value.y), abs(value.z), 1.0F};
  //});
  //std::vector<int> indices(points.size() / 2);
  //std::iota(indices.begin(), indices.end(), 0);
  //std::transform(indices.begin(), indices.end(), indices.begin(), [ ](const int& value)
  //{
  //  return 2 * value;
  //});
  //
  //glm::ivec4 viewport;
  //glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));
  //glm::ivec2 size(viewport[2] / 4, viewport[3] / 4);
  //
  //
  //// Setup renderer.
  //renderer_ = std::make_unique<ospray::cpp::Renderer>("scivis");
  //renderer_->set   ("shadowsEnabled", 1 );
  //renderer_->set   ("aoSamples"     , 8 );
  //renderer_->set   ("spp"           , 16);
  //renderer_->set   ("bgColor"       , 0.02F, 0.02F, 0.02F, 1.0F);
  //renderer_->commit();
  //
  //// Setup model.
  //streamlines_ = std::make_unique<ospray::cpp::Geometry>("streamlines");
  //vertex_data_ = std::make_unique<ospray::cpp::Data>(input_vertices    .size(), OSP_FLOAT3A, input_vertices    .data());
  //color_data_  = std::make_unique<ospray::cpp::Data>(input_colors.size(), OSP_FLOAT4 , input_colors.data());
  //index_data_  = std::make_unique<ospray::cpp::Data>(indices    .size(), OSP_INT    , indices    .data()); 
  //vertex_data_->commit();
  //color_data_ ->commit();
  //index_data_ ->commit();
  //
  //streamlines_->set("radius"      , 1.0F               );
  //streamlines_->set("vertex"      , *vertex_data_.get());
  //streamlines_->set("vertex.color", *color_data_ .get());
  //streamlines_->set("index"       , *index_data_ .get());
  //
  //auto material = renderer_->newMaterial("OBJMaterial");
  //material.set   ("Kd", 0.3F, 0.3F, 0.3F);
  //material.set   ("Ks", 0.5F, 0.5F, 0.5F);
  //material.set   ("Ns", 2.0F);
  //material.commit();
  //streamlines_->setMaterial(material);
  //streamlines_->commit     ();
  //
  //
  //model_ = std::make_unique<ospray::cpp::Model>();
  //model_   ->addGeometry(*streamlines_.get());
  //model_   ->commit     ();
  //renderer_->set        ("model", *model_.get());
  //renderer_->commit     ();
  //
  //// Setup camera.
  //camera_ = std::make_unique<ospray::cpp::Camera>("perspective");
  //camera_  ->set   ("aspect", size[0] / static_cast<float>(size[1]));
  //camera_  ->commit();
  //renderer_->set   ("camera", *camera_.get());
  //renderer_->commit();
  //
  //// Setup lights.
  //auto ambient_light = renderer_->newLight("ambient");
  //ambient_light.set   ("intensity"      , 0.2F);
  //ambient_light.commit();
  //
  //auto distant_light = renderer_->newLight("distant");
  //distant_light.set   ("direction"      , 1.0F, 1.0F, -0.5F);
  //distant_light.set   ("color"          , 1.0F, 1.0F,  0.8F);
  //distant_light.set   ("intensity"      , 0.8F);
  //distant_light.set   ("angularDiameter", 1.0F);
  //distant_light.commit();
  //
  //auto lights = std::vector<ospray::cpp::Light>{ambient_light, distant_light};
  //lights_ = std::make_unique<ospray::cpp::Data>(lights.size(), OSP_LIGHT, lights.data(), 0);
  //lights_  ->commit();
  //renderer_->set   ("lights", lights_.get());
  //renderer_->commit();
  //
  //// Setup framebuffer.
  //framebuffer_ = std::make_unique<ospray::cpp::FrameBuffer>(ospcommon::vec2i(size[0], size[1]), OSP_FB_SRGBA, OSP_FB_COLOR);
  //
  //auto camera_position = glm::vec3(0, 0, -100);
  //auto camera_forward  = glm::vec3(0, 0, 1);
  //auto camera_up       = glm::vec3(0, 1, 0);
  //camera_->set   ("aspect", size[0] / static_cast<float>(size[1]));
  //camera_->set   ("pos"   , camera_position[0], camera_position[1], camera_position[2]);
  //camera_->set   ("dir"   , camera_forward [0], camera_forward [1], camera_forward [2]);
  //camera_->set   ("up"    , camera_up      [0], camera_up      [1], camera_up      [2]);
  //camera_->commit();
  //
  //framebuffer_ = std::make_unique<ospray::cpp::FrameBuffer>(ospcommon::vec2i(size[0], size[1]), OSP_FB_SRGBA, OSP_FB_COLOR);
  //framebuffer_ ->clear      (OSP_FB_COLOR);
  //renderer_    ->renderFrame(*framebuffer_.get(), 0);
  //
  //const auto bytes = static_cast<uint32_t*>(framebuffer_->map(OSP_FB_COLOR));
  //
  //const auto file = fopen("temp.ppm", "wb");
  //fprintf(file, "P6\n%i %i\n255\n", size.x, size.y);
  //const auto out = static_cast<unsigned char *>(alloca(3 * size.x));
  //for (auto y = 0; y < size.y; y++) {
  //  const auto in = reinterpret_cast<const unsigned char *>(&bytes[(size.y - 1 - y) * size.x]);
  //  for (auto x = 0; x < size.x; x++) 
  //  {
  //    out[3 * x + 0] = in[4 * x + 0];
  //    out[3 * x + 1] = in[4 * x + 1];
  //    out[3 * x + 2] = in[4 * x + 2];
  //  }
  //  fwrite(out, 3 * size.x, sizeof(char), file);
  //}
  //fprintf(file, "\n");
  //fclose(file);
  //
  //framebuffer_ ->unmap      (bytes);
}
}