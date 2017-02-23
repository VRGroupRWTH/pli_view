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

  glDrawElements(GL_TRIANGLES, draw_count_, GL_UNSIGNED_INT, nullptr);

  index_buffer_  ->unbind();
  vertex_array_  ->unbind();
  shader_program_->unbind();
}

void odf_field::set_data(
  const uint3&   dimensions       ,
  const unsigned coefficient_count,
  const float*   coefficients     ,
  const uint2&   tessellations    , 
  const float3&  spacing          , 
  const uint3&   block_size       , 
  const float    scale            )
{
  auto voxel_count        = dimensions.x * dimensions.y * dimensions.z;
  auto tessellation_count = tessellations.x * tessellations.y;
  auto point_count        = voxel_count * tessellation_count;
  draw_count_             = 6 * point_count;
  
  vertex_buffer_->bind         ();
  vertex_buffer_->allocate     (point_count * sizeof(float3));
  vertex_buffer_->unbind       ();
  vertex_buffer_->cuda_register(cudaGraphicsMapFlagsNone);

  color_buffer_->bind          ();
  color_buffer_->allocate      (point_count * sizeof(float4));
  color_buffer_->unbind        ();
  color_buffer_->cuda_register (cudaGraphicsMapFlagsNone);
  
  index_buffer_ ->bind         ();
  index_buffer_ ->allocate     (draw_count_ * sizeof(unsigned));
  index_buffer_ ->unbind       ();
  index_buffer_ ->cuda_register(cudaGraphicsMapFlagsNone);

  auto cuda_vertex_buffer = vertex_buffer_->cuda_map<float3  >();
  auto cuda_color_buffer  = color_buffer_ ->cuda_map<float4  >();
  auto cuda_index_buffer  = index_buffer_ ->cuda_map<unsigned>();

  create_odfs(
    dimensions        ,
    coefficient_count ,
    coefficients      ,
    tessellations     ,
    spacing           ,
    block_size        ,
    scale             ,
    cuda_vertex_buffer,
    cuda_color_buffer ,
    cuda_index_buffer );

  index_buffer_ ->cuda_unmap();
  color_buffer_ ->cuda_unmap();
  vertex_buffer_->cuda_unmap();
}
}
