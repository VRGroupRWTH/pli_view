#include <pli_vis/visualization/algorithms/streamline_renderer.hpp>

#include <pli_vis/visualization/primitives/camera.hpp>
#include <shaders/streamline_renderer.vert.glsl>
#include <shaders/streamline_renderer.frag.glsl>

namespace pli
{
void streamline_renderer::initialize()
{
  program_         .reset(new gl::program     );
  vertex_array_    .reset(new gl::vertex_array);
  vertex_buffer_   .reset(new gl::array_buffer);
  direction_buffer_.reset(new gl::array_buffer);
  index_buffer_    .reset(new gl::index_buffer);

  program_         ->attach_shader         (gl::vertex_shader  (shaders::streamline_renderer_vert));
  program_         ->attach_shader         (gl::fragment_shader(shaders::streamline_renderer_frag));
  program_         ->link                  ();
  vertex_array_    ->bind                  ();
  program_         ->bind                  ();
  vertex_buffer_   ->bind                  ();
  program_         ->set_attribute_buffer  ("vertex"   , 4, GL_FLOAT);
  program_         ->enable_attribute_array("vertex");
  vertex_buffer_   ->unbind                ();
  direction_buffer_->bind                  ();
  program_         ->set_attribute_buffer  ("direction", 4, GL_FLOAT);
  program_         ->enable_attribute_array("direction");
  direction_buffer_->unbind                ();
  index_buffer_    ->bind                  ();
  vertex_array_    ->set_element_buffer    (*index_buffer_.get());
  index_buffer_    ->unbind                ();
  program_         ->unbind                ();
  vertex_array_    ->unbind                ();
}
void streamline_renderer::render    (const camera* camera)
{
  vertex_array_->bind       ();
  program_     ->bind       ();
  index_buffer_->bind       ();

  program_     ->set_uniform("color_mode"    , color_mode_);
  program_     ->set_uniform("color_k"       , color_k_);
  program_     ->set_uniform("color_inverted", color_inverted_);
  program_     ->set_uniform("model"         , absolute_matrix                ());
  program_     ->set_uniform("view"          , camera->inverse_absolute_matrix());
  program_     ->set_uniform("projection"    , camera->projection_matrix      ());

  glEnable                  (GL_LINE_SMOOTH);
  glHint                    (GL_LINE_SMOOTH_HINT, GL_NICEST);
  glEnable                  (GL_BLEND);
  glBlendFunc               (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDrawElements            (GL_LINES, GLsizei(draw_count_), GL_UNSIGNED_INT, nullptr);
  glDisable                 (GL_BLEND);
  glDisable                 (GL_LINE_SMOOTH);
  
  index_buffer_->unbind     ();
  program_     ->unbind     ();
  vertex_array_->unbind     ();
}

void streamline_renderer::set_data(
  const std::vector<float4>&   points    , 
  const std::vector<float4>&   directions,
  const std::vector<unsigned>& indices   )
{
  vertex_buffer_   ->bind    ();
  vertex_buffer_   ->set_data(points    .size() * sizeof(float4)  , points    .data());
  vertex_buffer_   ->unbind  ();
  
  direction_buffer_->bind    ();
  direction_buffer_->set_data(directions.size() * sizeof(float4)  , directions.data());
  direction_buffer_->unbind  ();

  index_buffer_    ->bind    ();
  index_buffer_    ->set_data(indices   .size() * sizeof(unsigned), indices   .data());
  index_buffer_    ->unbind  ();
  
  draw_count_ = indices.size();
}
}
