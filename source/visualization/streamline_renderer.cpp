#include <pli_vis/visualization/streamline_renderer.hpp>

#include <pli_vis/visualization/camera.hpp>
#include <shaders/simple_color.vert.glsl>
#include <shaders/simple_color.frag.glsl>

namespace pli
{
void streamline_renderer::set_data(const std::vector<float3>& points, const std::vector<float4>& colors)
{
  draw_count_ = points.size();
  
  vertex_buffer_->bind    ();
  vertex_buffer_->set_data(draw_count_ * sizeof(float3), points.data());
  vertex_buffer_->unbind  ();
  
  color_buffer_ ->bind    ();
  color_buffer_ ->set_data(draw_count_ * sizeof(float4), colors.data());
  color_buffer_ ->unbind  ();
}

void streamline_renderer::initialize()
{
  shader_program_.reset(new gl::program     );
  vertex_array_  .reset(new gl::vertex_array);
  vertex_buffer_ .reset(new gl::array_buffer);
  color_buffer_  .reset(new gl::array_buffer);
  index_buffer_  .reset(new gl::index_buffer);

  shader_program_->attach_shader(gl::vertex_shader  (shaders::simple_color_vert));
  shader_program_->attach_shader(gl::fragment_shader(shaders::simple_color_frag));
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
void streamline_renderer::render    (const camera* camera)
{
  shader_program_->bind();
  vertex_array_  ->bind();
  index_buffer_  ->bind();

  shader_program_->set_uniform("projection", camera->projection_matrix      ());
  shader_program_->set_uniform("view"      , camera->inverse_absolute_matrix());
  
  glEnable    (GL_LINE_SMOOTH);
  glHint      (GL_LINE_SMOOTH_HINT, GL_NICEST);

  glEnable    (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glDrawArrays(GL_LINES, 0, GLsizei(draw_count_));

  glDisable   (GL_BLEND);
  glDisable   (GL_LINE_SMOOTH);

  index_buffer_  ->unbind();
  vertex_array_  ->unbind();
  shader_program_->unbind();
}
}
