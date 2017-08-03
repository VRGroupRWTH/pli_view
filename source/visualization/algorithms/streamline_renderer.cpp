#include <pli_vis/visualization/algorithms/streamline_renderer.hpp>

#include <pli_vis/visualization/primitives/camera.hpp>
#include <shaders/depth_pass.vert.glsl>
#include <shaders/depth_pass.frag.glsl>
#include <shaders/view_dependent.vert.glsl>
#include <shaders/view_dependent.frag.glsl>

namespace pli
{
void streamline_renderer::initialize()
{
  GLint      default_framebuffer_id; glGetIntegerv(GL_FRAMEBUFFER_BINDING, &default_framebuffer_id);
  glm::ivec4 viewport;               glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));

  gl::framebuffer default_framebuffer(default_framebuffer_id);

  depth_pass_program_     .reset(new gl::program     );
  main_pass_program_      .reset(new gl::program     );
  depth_pass_vertex_array_.reset(new gl::vertex_array);
  main_pass_vertex_array_ .reset(new gl::vertex_array);
  vertex_buffer_          .reset(new gl::array_buffer);
  direction_buffer_       .reset(new gl::array_buffer);
  depth_framebuffer_      .reset(new gl::framebuffer );
  depth_texture_          .reset(new gl::texture_2d  );
  
  depth_pass_program_->attach_shader(gl::vertex_shader  (shaders::depth_pass_vert));
  depth_pass_program_->attach_shader(gl::fragment_shader(shaders::depth_pass_frag));
  depth_pass_program_->link();

  main_pass_program_->attach_shader(gl::vertex_shader  (shaders::view_dependent_vert));
  main_pass_program_->attach_shader(gl::fragment_shader(shaders::view_dependent_frag));
  main_pass_program_->link();
  
  depth_texture_->bind      ();
  depth_texture_->wrap_s    (GL_CLAMP_TO_EDGE);
  depth_texture_->wrap_t    (GL_CLAMP_TO_EDGE);
  depth_texture_->min_filter(GL_NEAREST);
  depth_texture_->mag_filter(GL_NEAREST);
  depth_texture_->set_image (GL_DEPTH_COMPONENT32F, viewport[2], viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT);
  depth_texture_->unbind    ();
  
  depth_framebuffer_->bind       ();
  depth_framebuffer_->set_texture(GL_DEPTH_ATTACHMENT , *depth_texture_.get());
  assert(depth_framebuffer_->is_valid() && depth_framebuffer_->is_complete());
  default_framebuffer.bind();
  
  depth_pass_vertex_array_->bind  ();
  depth_pass_program_     ->bind  ();
  vertex_buffer_          ->bind  ();
  depth_pass_program_     ->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  depth_pass_program_     ->enable_attribute_array("vertex");
  vertex_buffer_          ->unbind();
  depth_pass_program_     ->unbind();
  depth_pass_vertex_array_->unbind();

  main_pass_vertex_array_ ->bind  ();
  main_pass_program_      ->bind  ();
  vertex_buffer_          ->bind  ();
  main_pass_program_      ->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  main_pass_program_      ->enable_attribute_array("vertex");
  vertex_buffer_          ->unbind();
  direction_buffer_       ->bind  ();
  main_pass_program_      ->set_attribute_buffer  ("direction" , 3, GL_FLOAT);
  main_pass_program_      ->enable_attribute_array("direction");
  direction_buffer_       ->unbind();
  main_pass_program_      ->unbind();
  main_pass_vertex_array_ ->unbind();
}
void streamline_renderer::render    (const camera* camera)
{
  GLint default_framebuffer_id; glGetIntegerv(GL_FRAMEBUFFER_BINDING, &default_framebuffer_id);
  glm::ivec4 viewport;          glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));

  gl::framebuffer default_framebuffer(default_framebuffer_id);

  depth_texture_->bind     ();
  depth_texture_->set_image(GL_DEPTH_COMPONENT32F, viewport[2], viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT);
  depth_texture_->unbind   ();

  depth_framebuffer_      ->bind  ();
  depth_pass_vertex_array_->bind  ();
  depth_pass_program_     ->bind  ();
  depth_pass_program_     ->set_uniform("model"        , absolute_matrix                ());
  depth_pass_program_     ->set_uniform("view"         , camera->inverse_absolute_matrix());
  depth_pass_program_     ->set_uniform("projection"   , camera->projection_matrix      ());
  glEnable    (GL_DEPTH_TEST);
  glViewport  (viewport[0], viewport[1], viewport[2], viewport[3]);
  glClear     (GL_DEPTH_BUFFER_BIT);
  glEnable    (GL_LINE_SMOOTH);
  glHint      (GL_LINE_SMOOTH_HINT, GL_NICEST);
  glEnable    (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDrawArrays(GL_LINES, 0, GLsizei(draw_count_));
  glDisable   (GL_BLEND);
  glDisable   (GL_LINE_SMOOTH);
  glDisable   (GL_DEPTH_TEST);
  depth_pass_program_     ->unbind();
  depth_pass_vertex_array_->unbind();
  default_framebuffer     . bind  ();

  main_pass_vertex_array_->bind  ();
  main_pass_program_     ->bind  ();
  depth_texture_         ->set_active(0);
  depth_texture_         ->bind  ();
  main_pass_program_     ->set_uniform("depth_texture" , 0                                   );
  main_pass_program_     ->set_uniform("screen_size"   , glm::uvec2(viewport[2], viewport[3]));
  main_pass_program_     ->set_uniform("model"         , absolute_matrix                   ());
  main_pass_program_     ->set_uniform("view"          , camera->inverse_absolute_matrix   ());
  main_pass_program_     ->set_uniform("projection"    , camera->projection_matrix         ());
  main_pass_program_     ->set_uniform("view_dependent", view_dependent_transparency_        );
  main_pass_program_     ->set_uniform("rate_of_decay" , view_dependent_rate_of_decay_       );
  glEnable    (GL_LINE_SMOOTH);
  glHint      (GL_LINE_SMOOTH_HINT, GL_NICEST);
  glEnable    (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDrawArrays(GL_LINES, 0, GLsizei(draw_count_));
  glDisable   (GL_BLEND);
  glDisable   (GL_LINE_SMOOTH);
  depth_texture_         ->unbind();
  main_pass_program_     ->unbind();
  main_pass_vertex_array_->unbind();
}
  
void streamline_renderer::set_data(const std::vector<float3>& points, const std::vector<float3>& directions)
{
  draw_count_ = points.size();
  
  vertex_buffer_   ->bind    ();
  vertex_buffer_   ->set_data(draw_count_ * sizeof(float3), points    .data());
  vertex_buffer_   ->unbind  ();
  
  direction_buffer_->bind    ();
  direction_buffer_->set_data(draw_count_ * sizeof(float3), directions.data());
  direction_buffer_->unbind  ();
}
void streamline_renderer::set_view_dependent_transparency (bool  enabled)
{
  view_dependent_transparency_  = enabled;
}
void streamline_renderer::set_view_dependent_rate_of_decay(float value  )
{
  view_dependent_rate_of_decay_ = value  ;
}
}
