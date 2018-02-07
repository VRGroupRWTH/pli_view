#include <pli_vis/visualization/algorithms/polar_plot_field.hpp>

#include <pli_vis/visualization/primitives/camera.hpp>
#include <shaders/polar_plot.vert.glsl>
#include <shaders/polar_plot.frag.glsl>

namespace pli
{
void polar_plot_field::initialize()
{
  shader_program_  .reset(new gl::program     );
  vertex_array_    .reset(new gl::vertex_array);
  vertex_buffer_   .reset(new gl::array_buffer);
  direction_buffer_.reset(new gl::array_buffer);
  
  shader_program_  ->attach_shader         (gl::vertex_shader  (shaders::polar_plot_vert));
  shader_program_  ->attach_shader         (gl::fragment_shader(shaders::polar_plot_frag));
  shader_program_  ->link                  ();
                   
  shader_program_  ->bind                  ();
  vertex_array_    ->bind                  ();
                   
  vertex_buffer_   ->bind                  ();
  shader_program_  ->set_attribute_buffer  ("vertex"   , 3, GL_FLOAT);
  shader_program_  ->enable_attribute_array("vertex");
  vertex_buffer_   ->unbind                ();
  
  direction_buffer_->bind                  ();
  shader_program_  ->set_attribute_buffer  ("direction", 3, GL_FLOAT);
  shader_program_  ->enable_attribute_array("direction");
  direction_buffer_->unbind                ();

  vertex_array_    ->unbind                ();
  shader_program_  ->unbind                ();
}
void polar_plot_field::render    (const camera* camera)
{
  shader_program_  ->bind                  ();
  vertex_array_    ->bind                  ();
  shader_program_  ->set_uniform           ("color_mode"    , color_mode_    );
  shader_program_  ->set_uniform           ("color_k"       , color_k_       );
  shader_program_  ->set_uniform           ("color_inverted", color_inverted_);
  shader_program_  ->set_uniform           ("model"         , absolute_matrix                ());
  shader_program_  ->set_uniform           ("view"          , camera->inverse_absolute_matrix());
  shader_program_  ->set_uniform           ("projection"    , camera->projection_matrix      ());
  glDrawArrays                             (GL_TRIANGLES    , 0, GLsizei(draw_count_));
  vertex_array_    ->unbind                ();
  shader_program_  ->unbind                ();
}

void polar_plot_field::set_data(const std::vector<float3>& vertices, const std::vector<float3>& directions)
{        
  vertex_buffer_   ->bind                  ();
  vertex_buffer_   ->set_data              (vertices  .size() * sizeof float3, vertices  .data());
  vertex_buffer_   ->unbind                ();
  
  direction_buffer_->bind                  ();
  direction_buffer_->set_data              (directions.size() * sizeof float3, directions.data());
  direction_buffer_->unbind                ();

  draw_count_ = vertices.size();
}
}
