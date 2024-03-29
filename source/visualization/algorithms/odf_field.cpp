#include <pli_vis/visualization/algorithms/odf_field.hpp>

#include <algorithm>

#include <pli_vis/cuda/odf_field.h>
#include <pli_vis/visualization/primitives/camera.hpp>
#include <shaders/odf_field_renderer.vert.glsl>
#include <shaders/odf_field_renderer.frag.glsl>

namespace pli
{
void odf_field::initialize()
{
  shader_program_.reset(new gl::program     );
  vertex_array_  .reset(new gl::vertex_array);
  vertex_buffer_ .reset(new gl::array_buffer);
  color_buffer_  .reset(new gl::array_buffer);
  index_buffer_  .reset(new gl::index_buffer);

  shader_program_->attach_shader(gl::vertex_shader  (shaders::odf_field_renderer_vert));
  shader_program_->attach_shader(gl::fragment_shader(shaders::odf_field_renderer_frag));
  shader_program_->link();
  
  shader_program_->bind();
  vertex_array_  ->bind();

  vertex_buffer_ ->bind();
  shader_program_->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  shader_program_->enable_attribute_array("vertex");
  vertex_buffer_ ->unbind();

  color_buffer_  ->bind();
  shader_program_->set_attribute_buffer  ("direction" , 3, GL_FLOAT);
  shader_program_->enable_attribute_array("direction");
  color_buffer_  ->unbind();

  vertex_array_  ->unbind();
  shader_program_->unbind();
}
void odf_field::render    (const camera* camera)
{
  shader_program_->bind();
  vertex_array_  ->bind();
  index_buffer_  ->bind();

  shader_program_->set_uniform("color_mode"    , color_mode_);
  shader_program_->set_uniform("color_k"       , color_k_);
  shader_program_->set_uniform("color_inverted", color_inverted_);
  shader_program_->set_uniform("model"         , absolute_matrix                ());
  shader_program_->set_uniform("view"          , camera->inverse_absolute_matrix());
  shader_program_->set_uniform("projection"    , camera->projection_matrix      ());

  // Select by visible layers.
  auto dimension_count  = dimensions_.z > 1 ? 3 : 2;
  auto min_dimension    = std::min(dimensions_.x, dimensions_.y);
  if (dimension_count == 3)
    min_dimension = std::min(min_dimension, dimensions_.z);
  auto max_layer        = int(log(min_dimension) / log(2));
  auto layer_offset     = 0;
  auto layer_dimensions = dimensions_;
  auto indices_count    = 6 * tessellations_.x * tessellations_.y;
  for (auto layer = max_layer; layer >= 0; layer--)
  {
    if (visible_layers_[layer])
    {
      auto layer_indices_offset =
        layer_offset *
        indices_count;
      auto layer_indices_count  =
        layer_dimensions.x *
        layer_dimensions.y *
        layer_dimensions.z *
        indices_count;
      glDrawElements(GL_TRIANGLES, layer_indices_count, GL_UNSIGNED_INT, (void*) (layer_indices_offset * sizeof(GLuint)));
    }

    layer_offset += 
      layer_dimensions.x * 
      layer_dimensions.y * 
      layer_dimensions.z;
    layer_dimensions = {
      layer_dimensions.x / 2,
      layer_dimensions.y / 2,
      dimension_count == 3 ? layer_dimensions.z / 2 : 1
    };
  }

  index_buffer_  ->unbind();
  vertex_array_  ->unbind();
  shader_program_->unbind();
}

void odf_field::set_data(
  const uint3&   dimensions       ,
  const unsigned maximum_degree   ,
  const float*   coefficients     ,
  const uint2&   tessellations    ,
  const uint3&   vector_dimensions, 
  const float    scale            ,
  const bool     hierarchical     ,
  const bool     clustering       ,
  const float    cluster_threshold,
  std::function<void(const std::string&)> status_callback)
{
  dimensions_    = dimensions;
  tessellations_ = tessellations;

  auto voxel_count = dimensions_.x * dimensions_.y * dimensions_.z;
  if(hierarchical)
  {
    auto dimension_count  = dimensions_.z > 1 ? 3 : 2;
    auto min_dimension    = std::min(dimensions_.x, dimensions_.y);
    if (dimension_count == 3)
      min_dimension       = std::min(min_dimension, dimensions_.z);
    auto max_layer        = int(log(min_dimension) / log(2));
    voxel_count           = unsigned(voxel_count * 
      ((1.0 - pow(1.0 / pow(2, dimension_count), max_layer + 1)) / 
       (1.0 -     1.0 / pow(2, dimension_count))));
  }

  auto tessellation_count = tessellations.x * tessellations.y;
  auto point_count        = voxel_count * tessellation_count;
  draw_count_             = 6 * point_count;
  
  vertex_buffer_->bind           ();
  vertex_buffer_->allocate       (point_count * sizeof(float3));
  vertex_buffer_->unbind         ();
  vertex_buffer_->cuda_register  (cudaGraphicsMapFlagsNone);

  color_buffer_->bind           ();
  color_buffer_->allocate       (point_count * sizeof(float3));
  color_buffer_->unbind         ();
  color_buffer_->cuda_register  (cudaGraphicsMapFlagsNone);

  index_buffer_ ->bind           ();
  index_buffer_ ->allocate       (draw_count_ * sizeof(unsigned));
  index_buffer_ ->unbind         ();
  index_buffer_ ->cuda_register  (cudaGraphicsMapFlagsNone);

  auto cuda_vertex_buffer = vertex_buffer_->cuda_map<float3  >();
  auto cuda_color_buffer  = color_buffer_ ->cuda_map<float3  >();
  auto cuda_index_buffer  = index_buffer_ ->cuda_map<unsigned>();

  sample_odfs(
    dimensions_       ,
    maximum_degree    ,
    coefficients      ,
    tessellations     ,
    vector_dimensions ,
    scale             ,
    cuda_vertex_buffer,
    cuda_color_buffer ,
    cuda_index_buffer ,
    hierarchical      ,
    clustering        ,
    cluster_threshold ,
    status_callback   );

  index_buffer_ ->cuda_unmap();
  color_buffer_ ->cuda_unmap();
  vertex_buffer_->cuda_unmap();
  
  index_buffer_ ->cuda_unregister();
  color_buffer_ ->cuda_unregister();
  vertex_buffer_->cuda_unregister();
}

void odf_field::set_visible_layers(const std::vector<bool>& visible_layers)
{
  visible_layers_ = visible_layers;
}
}
