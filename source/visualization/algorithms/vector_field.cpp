#include <pli_vis/visualization/algorithms/vector_field.hpp>

#include <pli_vis/visualization/primitives/camera.hpp>
#include <shaders/view_dependent_vector_field.vert.glsl>
#include <shaders/view_dependent_vector_field.geom.glsl>
#include <shaders/view_dependent_vector_field.frag.glsl>

namespace pli
{
void vector_field::initialize()
{
  shader_program_  .reset(new gl::program     );
  vertex_array_    .reset(new gl::vertex_array);
  direction_buffer_.reset(new gl::array_buffer);

  shader_program_->attach_shader(gl::vertex_shader  (shaders::view_dependent_vector_field_vert));
  shader_program_->attach_shader(gl::geometry_shader(shaders::view_dependent_vector_field_geom));
  shader_program_->attach_shader(gl::fragment_shader(shaders::view_dependent_vector_field_frag));
  shader_program_->link();
  
  shader_program_->bind();
  vertex_array_  ->bind();

  direction_buffer_->bind();
  shader_program_  ->set_attribute_buffer  ("direction", 3, GL_FLOAT);
  shader_program_  ->enable_attribute_array("direction");
  direction_buffer_->unbind();

  vertex_array_  ->unbind();
  shader_program_->unbind();
}
void vector_field::render    (const camera* camera)
{
  shader_program_->bind  ();
  vertex_array_  ->bind  ();
  
  shader_program_->set_uniform("color_mode"       , color_mode_);
  shader_program_->set_uniform("color_k"          , color_k_);
  shader_program_->set_uniform("color_inverted"   , color_inverted_);
  shader_program_->set_uniform("model"            , absolute_matrix                ());
  shader_program_->set_uniform("view"             , camera->inverse_absolute_matrix());
  shader_program_->set_uniform("projection"       , camera->projection_matrix      ());
  shader_program_->set_uniform("dimensions"       , dimensions_);
  shader_program_->set_uniform("vectors_per_point", vectors_per_point_);
  shader_program_->set_uniform("scale"            , scale_);
  shader_program_->set_uniform("view_dependent"   , view_dependent_transparency_ );
  shader_program_->set_uniform("rate_of_decay"    , view_dependent_rate_of_decay_);
  
  glEnable    (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDrawArrays(GL_POINTS, 0, GLsizei(draw_count_));
  glDisable   (GL_BLEND);

  vertex_array_  ->unbind();
  shader_program_->unbind();
}

void vector_field::set_data(
  const uint3&   dimensions       ,
  const unsigned vectors_per_point,
  const float3*  unit_vectors     ,
  std::function<void(const std::string&)> status_callback)
{
  dimensions_        = glm::uvec3(dimensions.x, dimensions.y, dimensions.z);
  vectors_per_point_ = vectors_per_point;
  draw_count_        = dimensions.x * dimensions.y * dimensions.z * vectors_per_point;

  direction_buffer_->bind    ();
  direction_buffer_->set_data(draw_count_ * sizeof(float3), unit_vectors);
  direction_buffer_->unbind  ();
}

void vector_field::set_scale                       (float    scale            )
{
  scale_ = scale;
}
void vector_field::set_view_dependent_transparency (bool     enabled          )
{
  view_dependent_transparency_  = enabled;
}
void vector_field::set_view_dependent_rate_of_decay(float    value            )
{
  view_dependent_rate_of_decay_ = value  ;
}
}

