#include <pli_vis/visualization/algorithms/vector_field.hpp>

#include <pli_vis/cuda/vector_field.h>
#include <pli_vis/visualization/primitives/camera.hpp>
#include <shaders/view_dependent.vert.glsl>
#include <shaders/view_dependent.frag.glsl>

namespace pli
{
void vector_field::initialize()
{
  shader_program_.reset(new gl::program     );
  vertex_array_  .reset(new gl::vertex_array);
  vertex_buffer_ .reset(new gl::array_buffer);
  color_buffer_  .reset(new gl::array_buffer);

  shader_program_->attach_shader(gl::vertex_shader  (shaders::view_dependent_vert));
  shader_program_->attach_shader(gl::fragment_shader(shaders::view_dependent_frag));
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
void vector_field::render    (const camera* camera)
{
  shader_program_->bind  ();
  vertex_array_  ->bind  ();

  shader_program_->set_uniform("projection"    , camera->projection_matrix      ());
  shader_program_->set_uniform("view"          , camera->inverse_absolute_matrix());
  shader_program_->set_uniform("view_dependent", view_dependent_transparency_ );
  shader_program_->set_uniform("rate_of_decay" , view_dependent_rate_of_decay_);
  
  glEnable    (GL_LINE_SMOOTH);
  glHint      (GL_LINE_SMOOTH_HINT, GL_NICEST);

  glEnable    (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glDrawArrays(GL_LINES, 0, GLsizei(draw_count_));
  
  glDisable   (GL_BLEND);
  glDisable   (GL_LINE_SMOOTH);

  vertex_array_  ->unbind();
  shader_program_->unbind();
}

void vector_field::set_data(
  const uint3&  dimensions  ,
  const float3* unit_vectors,
  float         scale       ,
  std::function<void(const std::string&)> status_callback)
{
  draw_count_ = 2 * dimensions.x * dimensions.y * dimensions.z;

  vertex_buffer_->bind         ();
  vertex_buffer_->allocate     (draw_count_ * sizeof(float3));
  vertex_buffer_->unbind       ();
  vertex_buffer_->cuda_register();

  color_buffer_ ->bind         ();
  color_buffer_ ->allocate     (draw_count_ * sizeof(float4));
  color_buffer_ ->unbind       ();
  color_buffer_ ->cuda_register();

  auto cuda_vertex_buffer = vertex_buffer_->cuda_map<float3>();
  auto cuda_color_buffer  = color_buffer_ ->cuda_map<float4>();
  create_vector_field(
    dimensions        ,
    unit_vectors      ,
    scale             ,
    cuda_vertex_buffer,
    cuda_color_buffer ,
    status_callback   );

  color_buffer_ ->cuda_unmap();
  vertex_buffer_->cuda_unmap();
}

void vector_field::set_view_dependent_transparency (bool  enabled)
{
  view_dependent_transparency_ = enabled;
}
void vector_field::set_view_dependent_rate_of_decay(float value  )
{
  view_dependent_rate_of_decay_ = value;
}
}

