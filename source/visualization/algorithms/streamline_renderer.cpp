#include <pli_vis/visualization/algorithms/streamline_renderer.hpp>

#include <pli_vis/visualization/primitives/camera.hpp>
#include <shaders/lineao_color_pass.vert.glsl>
#include <shaders/lineao_color_pass.frag.glsl>
#include <shaders/lineao_normal_depth_pass.vert.glsl>
#include <shaders/lineao_normal_depth_pass.frag.glsl>
#include <shaders/lineao_zoom_pass.vert.glsl>
#include <shaders/lineao_zoom_pass.frag.glsl>
#include <shaders/view_dependent.vert.glsl>
#include <shaders/view_dependent.frag.glsl>

namespace pli
{
void streamline_renderer::initialize()
{
  glm::ivec4 viewport;
  glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));
  glm::uvec2 screen_size {viewport.z, viewport.w};
  
  vertex_buffer_   .reset(new gl::array_buffer);
  direction_buffer_.reset(new gl::array_buffer);

  initialize_normal_depth_pass(screen_size);
  initialize_color_pass       (screen_size);
  initialize_zoom_pass        (screen_size);
  initialize_main_pass        ();
}
void streamline_renderer::render    (const camera* camera)
{
  glm::ivec4 viewport;
  glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));
  glm::uvec2 screen_size {viewport.z, viewport.w};
  
  render_normal_depth_pass(camera, screen_size);
  render_color_pass       (camera, screen_size);
  render_zoom_pass        (camera, screen_size);
  render_main_pass        (camera, screen_size);
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

void streamline_renderer::initialize_normal_depth_pass(const glm::uvec2& screen_size)
{
  normal_depth_program_     .reset(new gl::program     );
  normal_depth_vertex_array_.reset(new gl::vertex_array);
  normal_depth_map_         .reset(new render_target(screen_size));
  
  normal_depth_program_->attach_shader(gl::vertex_shader  (shaders::lineao_normal_depth_pass_vert));
  normal_depth_program_->attach_shader(gl::fragment_shader(shaders::lineao_normal_depth_pass_frag));
  normal_depth_program_->link();
  
  normal_depth_vertex_array_->bind  ();
  normal_depth_program_     ->bind  ();
  vertex_buffer_            ->bind  ();
  normal_depth_program_     ->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  normal_depth_program_     ->enable_attribute_array("vertex");
  vertex_buffer_            ->unbind();
  direction_buffer_         ->bind  ();
  normal_depth_program_     ->set_attribute_buffer  ("direction" , 3, GL_FLOAT);
  normal_depth_program_     ->enable_attribute_array("direction");
  direction_buffer_         ->unbind();
  normal_depth_program_     ->unbind();
  normal_depth_vertex_array_->unbind();
}
void streamline_renderer::initialize_color_pass       (const glm::uvec2& screen_size)
{
  color_program_     .reset(new gl::program     );
  color_vertex_array_.reset(new gl::vertex_array);
  color_map_         .reset(new render_target(screen_size));
  
  color_program_->attach_shader(gl::vertex_shader  (shaders::lineao_color_pass_vert));
  color_program_->attach_shader(gl::fragment_shader(shaders::lineao_color_pass_frag));
  color_program_->link();
  
  color_vertex_array_->bind  ();
  color_program_     ->bind  ();
  vertex_buffer_     ->bind  ();
  color_program_     ->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  color_program_     ->enable_attribute_array("vertex");
  vertex_buffer_     ->unbind();
  direction_buffer_  ->bind  ();
  color_program_     ->set_attribute_buffer  ("direction" , 3, GL_FLOAT);
  color_program_     ->enable_attribute_array("direction");
  direction_buffer_  ->unbind();
  color_program_     ->unbind();
  color_vertex_array_->unbind();
}
void streamline_renderer::initialize_zoom_pass        (const glm::uvec2& screen_size)
{
  zoom_program_     .reset(new gl::program     );
  zoom_vertex_array_.reset(new gl::vertex_array);
  zoom_map_         .reset(new render_target(screen_size));
  
  zoom_program_->attach_shader(gl::vertex_shader  (shaders::lineao_zoom_pass_vert));
  zoom_program_->attach_shader(gl::fragment_shader(shaders::lineao_zoom_pass_frag));
  zoom_program_->link();
  
  zoom_vertex_array_->bind  ();
  zoom_program_     ->bind  ();
  vertex_buffer_    ->bind  ();
  zoom_program_     ->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  zoom_program_     ->enable_attribute_array("vertex");
  vertex_buffer_    ->unbind();
  direction_buffer_ ->bind  ();
  zoom_program_     ->set_attribute_buffer  ("direction" , 3, GL_FLOAT);
  zoom_program_     ->enable_attribute_array("direction");
  direction_buffer_ ->unbind();
  zoom_program_     ->unbind();
  zoom_vertex_array_->unbind();
}
void streamline_renderer::initialize_main_pass        ()
{
  program_         .reset(new gl::program     );
  vertex_array_    .reset(new gl::vertex_array);

  program_->attach_shader(gl::vertex_shader  (shaders::view_dependent_vert));
  program_->attach_shader(gl::fragment_shader(shaders::view_dependent_frag));
  program_->link();
  
  vertex_array_    ->bind  ();
  program_         ->bind  ();
  vertex_buffer_   ->bind  ();
  program_         ->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  program_         ->enable_attribute_array("vertex");
  vertex_buffer_   ->unbind();
  direction_buffer_->bind  ();
  program_         ->set_attribute_buffer  ("direction" , 3, GL_FLOAT);
  program_         ->enable_attribute_array("direction");
  direction_buffer_->unbind();
  program_         ->unbind();
  vertex_array_    ->unbind();
}

void streamline_renderer::render_normal_depth_pass(const camera* camera, const glm::uvec2& screen_size) const
{
  normal_depth_map_->bind();

  normal_depth_vertex_array_->bind  ();
  normal_depth_program_     ->bind  ();
  normal_depth_program_     ->set_uniform("screen_size", screen_size                      );
  normal_depth_program_     ->set_uniform("model"      , absolute_matrix                ());
  normal_depth_program_     ->set_uniform("view"       , camera->inverse_absolute_matrix());
  normal_depth_program_     ->set_uniform("projection" , camera->projection_matrix      ());

  glEnable    (GL_LINE_SMOOTH);
  glHint      (GL_LINE_SMOOTH_HINT, GL_NICEST);
  glEnable    (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDrawArrays(GL_LINES, 0, GLsizei(draw_count_));
  glDisable   (GL_BLEND);
  glDisable   (GL_LINE_SMOOTH);
  
  normal_depth_program_     ->unbind();
  normal_depth_vertex_array_->unbind();

  normal_depth_map_->unbind();
}
void streamline_renderer::render_color_pass       (const camera* camera, const glm::uvec2& screen_size) const
{
  color_map_->bind();

  color_vertex_array_->bind  ();
  color_program_     ->bind  ();
  color_program_     ->set_uniform("model"     , absolute_matrix                ());
  color_program_     ->set_uniform("view"      , camera->inverse_absolute_matrix());
  color_program_     ->set_uniform("projection", camera->projection_matrix      ());

  glEnable    (GL_LINE_SMOOTH);
  glHint      (GL_LINE_SMOOTH_HINT, GL_NICEST);
  glEnable    (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDrawArrays(GL_LINES, 0, GLsizei(draw_count_));
  glDisable   (GL_BLEND);
  glDisable   (GL_LINE_SMOOTH);
  
  color_program_     ->unbind();
  color_vertex_array_->unbind();

  color_map_->unbind();
}
void streamline_renderer::render_zoom_pass        (const camera* camera, const glm::uvec2& screen_size) const
{
  zoom_map_->bind();

  zoom_vertex_array_->bind  ();
  zoom_program_     ->bind  ();
  zoom_program_     ->set_uniform("model"     , absolute_matrix                ());
  zoom_program_     ->set_uniform("view"      , camera->inverse_absolute_matrix());
  zoom_program_     ->set_uniform("projection", camera->projection_matrix      ());

  glEnable    (GL_LINE_SMOOTH);
  glHint      (GL_LINE_SMOOTH_HINT, GL_NICEST);
  glEnable    (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDrawArrays(GL_LINES, 0, GLsizei(draw_count_));
  glDisable   (GL_BLEND);
  glDisable   (GL_LINE_SMOOTH);
  
  zoom_program_     ->unbind();
  zoom_vertex_array_->unbind();

  zoom_map_->unbind();
}
void streamline_renderer::render_main_pass        (const camera* camera, const glm::uvec2& screen_size) const
{
  vertex_array_->bind  ();
  program_     ->bind  ();
  program_     ->set_uniform("screen_size"   , screen_size                         );
  program_     ->set_uniform("model"         , absolute_matrix                   ());
  program_     ->set_uniform("view"          , camera->inverse_absolute_matrix   ());
  program_     ->set_uniform("projection"    , camera->projection_matrix         ());
  program_     ->set_uniform("view_dependent", view_dependent_transparency_        );
  program_     ->set_uniform("rate_of_decay" , view_dependent_rate_of_decay_       );
  
  auto nd_tex    = normal_depth_map_->color_texture();
  auto color_tex = color_map_       ->color_texture();
  auto zoom_tex  = zoom_map_        ->color_texture();
  nd_tex   ->set_active(0); nd_tex   ->bind(); program_->set_uniform("normal_depth_texture", 0);
  color_tex->set_active(1); color_tex->bind(); program_->set_uniform("color_texture"       , 1);
  zoom_tex ->set_active(2); zoom_tex ->bind(); program_->set_uniform("zoom_texture"        , 2);
  zoom_tex ->set_active(0);

  glEnable    (GL_LINE_SMOOTH);
  glHint      (GL_LINE_SMOOTH_HINT, GL_NICEST);
  glEnable    (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDrawArrays(GL_LINES, 0, GLsizei(draw_count_));
  glDisable   (GL_BLEND);
  glDisable   (GL_LINE_SMOOTH);
  
  program_     ->unbind();
  vertex_array_->unbind();
}
}
