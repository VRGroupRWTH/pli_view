#include /* implements */ <visualization/basic_tracer.hpp>

#include <shaders/vector_field.vert.glsl>
#include <shaders/vector_field.frag.glsl>
#include <math/camera.hpp>

namespace pli
{
void basic_tracer::trace(const boost::multi_array<float, 4>& vectors)
{
  // TODO: Run tangent, create connected lines, pass to GL.

  //linear_tracer tracer;
  //tracer.SetData(vectors.data());
  //tracer.
}

void basic_tracer::initialize()
{
  shader_program_.reset(new gl::program     );
  vertex_array_  .reset(new gl::vertex_array);
  vertex_buffer_ .reset(new gl::array_buffer);
  color_buffer_  .reset(new gl::array_buffer);
  index_buffer_  .reset(new gl::index_buffer);

  shader_program_->attach_shader(gl::vertex_shader  (shaders::vector_field_vert));
  shader_program_->attach_shader(gl::fragment_shader(shaders::vector_field_frag));
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
void basic_tracer::render    (const camera* camera)
{
  shader_program_->bind();
  vertex_array_  ->bind();
  index_buffer_  ->bind();

  shader_program_->set_uniform("projection", camera->projection_matrix      ());
  shader_program_->set_uniform("view"      , camera->inverse_absolute_matrix());

  glDrawArrays(GL_LINES, 0, GLsizei(draw_count_));

  index_buffer_  ->unbind();
  vertex_array_  ->unbind();
  shader_program_->unbind();
}
}
