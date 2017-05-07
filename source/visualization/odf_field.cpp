#include /* implements */ <visualization/odf_field.hpp>

#include <cuda/odf_field.h>
#include <math/camera.hpp>
#include <shaders/odf_field.vert.glsl>
#include <shaders/odf_field.frag.glsl>

namespace pli
{
void odf_field::initialize()
{
  shader_program_.reset(new gl::program     );
  vertex_array_  .reset(new gl::vertex_array);
  vertex_buffer_ .reset(new gl::array_buffer);
  color_buffer_  .reset(new gl::array_buffer);
  index_buffer_  .reset(new gl::index_buffer);

  shader_program_->attach_shader(gl::vertex_shader  (shaders::odf_field_vert));
  shader_program_->attach_shader(gl::fragment_shader(shaders::odf_field_frag));
  shader_program_->link();
  
  shader_program_->bind();
  vertex_array_  ->bind();

  vertex_buffer_ ->bind();
  shader_program_->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  shader_program_->enable_attribute_array("vertex");
  vertex_buffer_ ->unbind();

  color_buffer_  ->bind();
  shader_program_->set_attribute_buffer  ("color" , 4, GL_FLOAT);
  shader_program_->enable_attribute_array("color");
  color_buffer_  ->unbind();

  vertex_array_  ->unbind();
  shader_program_->unbind();
}
void odf_field::render    (const camera* camera)
{
  shader_program_->bind();
  vertex_array_  ->bind();
  index_buffer_  ->bind();

  shader_program_->set_uniform("projection", camera->projection_matrix      ());
  shader_program_->set_uniform("view"      , camera->inverse_absolute_matrix());

  // Select by visible layers.
  auto dimension_count  = dimensions_.z > 1 ? 3 : 2;
  auto min_dimension    = min(dimensions_.x, dimensions_.y);
  if (dimension_count == 3)
    min_dimension = min(min_dimension, dimensions_.z);
  auto max_layer        = log(min_dimension) / log(2);
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
  const unsigned coefficient_count,
  const float*   coefficients     ,
  const uint2&   tessellations    , 
  const float3&  vector_spacing   , 
  const uint3&   vector_dimensions, 
  const float    scale            ,
  const bool     clustering       ,
  const float    cluster_threshold,
  std::function<void(const std::string&)> status_callback)
{
  dimensions_    = dimensions;
  tessellations_ = tessellations;

  auto base_voxel_count = dimensions_.x * dimensions_.y * dimensions_.z;
  auto dimension_count  = dimensions_.z > 1 ? 3 : 2;
  auto min_dimension    = min(dimensions_.x, dimensions_.y);
  if (dimension_count == 3)
    min_dimension = min(min_dimension, dimensions_.z);
  auto max_layer        = log(min_dimension) / log(2);
  auto voxel_count      = unsigned(base_voxel_count * 
    ((1.0 - pow(1.0 / pow(2, dimension_count), max_layer + 1)) / 
     (1.0 -     1.0 / pow(2, dimension_count))));

  auto tessellation_count = tessellations.x * tessellations.y;
  auto point_count        = voxel_count * tessellation_count;
  draw_count_             = 6 * point_count;
  
  vertex_buffer_->bind           ();
  vertex_buffer_->allocate       (point_count * sizeof(float3));
  vertex_buffer_->unbind         ();
  vertex_buffer_->cuda_register  (cudaGraphicsMapFlagsNone);

  color_buffer_->bind           ();
  color_buffer_->allocate       (point_count * sizeof(float4));
  color_buffer_->unbind         ();
  color_buffer_->cuda_register  (cudaGraphicsMapFlagsNone);

  index_buffer_ ->bind           ();
  index_buffer_ ->allocate       (draw_count_ * sizeof(unsigned));
  index_buffer_ ->unbind         ();
  index_buffer_ ->cuda_register  (cudaGraphicsMapFlagsNone);

  auto cuda_vertex_buffer = vertex_buffer_->cuda_map<float3  >();
  auto cuda_color_buffer  = color_buffer_ ->cuda_map<float4  >();
  auto cuda_index_buffer  = index_buffer_ ->cuda_map<unsigned>();

  sample_odfs(
    dimensions_       ,
    coefficient_count ,
    coefficients      ,
    tessellations     ,
    vector_spacing    ,
    vector_dimensions ,
    scale             ,
    cuda_vertex_buffer,
    cuda_color_buffer ,
    cuda_index_buffer ,
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

void odf_field::set_visible_layers(
  const std::vector<bool>& visible_layers)
{
  visible_layers_ = visible_layers;
}
}
