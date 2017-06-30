#include /* implements */ <visualization/volume_renderer.hpp>

#include <math/camera.hpp>
#include <shaders/volume_renderer_prepass.vert.glsl>
#include <shaders/volume_renderer_prepass.frag.glsl>
#include <shaders/volume_renderer.vert.glsl>
#include <shaders/volume_renderer.frag.glsl>

namespace pli
{
void volume_renderer::initialize()
{
  prepass_shader_program_   .reset(new gl::program     );
  shader_program_           .reset(new gl::program     );
  prepass_vertex_array_     .reset(new gl::vertex_array);
  vertex_array_             .reset(new gl::vertex_array);
  vertex_buffer_            .reset(new gl::array_buffer);
  color_buffer_             .reset(new gl::array_buffer);
  index_buffer_             .reset(new gl::index_buffer);
  transfer_function_texture_.reset(new gl::texture_1d  );
  volume_texture_           .reset(new gl::texture_3d  );
  framebuffer_              .reset(new gl::framebuffer );
  exit_points_color_texture_.reset(new gl::texture_2d  );
  exit_points_depth_texture_.reset(new gl::texture_2d  );

  prepass_shader_program_->attach_shader(gl::vertex_shader  (shaders::volume_renderer_prepass_vert));
  prepass_shader_program_->attach_shader(gl::fragment_shader(shaders::volume_renderer_prepass_frag));
  prepass_shader_program_->link();

  shader_program_        ->attach_shader(gl::vertex_shader  (shaders::volume_renderer_vert));
  shader_program_        ->attach_shader(gl::fragment_shader(shaders::volume_renderer_frag));
  shader_program_        ->link();
  
  // Initialize prepass shader program and vertex array.
  prepass_shader_program_->bind();
  prepass_vertex_array_  ->bind();
                         
  vertex_buffer_         ->bind();
  prepass_shader_program_->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  prepass_shader_program_->enable_attribute_array("vertex");
  vertex_buffer_         ->unbind();
                
  prepass_vertex_array_  ->unbind();
  prepass_shader_program_->unbind();

  // Initialize main pass shader program and vertex array.
  shader_program_        ->bind();
  vertex_array_          ->bind();
                         
  vertex_buffer_         ->bind();
  shader_program_        ->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  shader_program_        ->enable_attribute_array("vertex");
  vertex_buffer_         ->unbind();
             
  vertex_array_          ->unbind();
  shader_program_        ->unbind();
  
  // Setup the framebuffer.
  framebuffer_->bind       ();
  framebuffer_->set_texture(GL_COLOR_ATTACHMENT0, *exit_points_color_texture_.get());
  framebuffer_->set_texture(GL_DEPTH_ATTACHMENT , *exit_points_depth_texture_.get());
  framebuffer_->unbind     ();

  // Set data (unit cube).
  std::vector<float>    vertices = {
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 
    0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    1.0, 1.0, 0.0, 1.0, 1.0, 1.0
  };
  std::vector<unsigned> indices  = {
    1, 5, 7, 7, 3, 1, 0, 2, 6, 6, 4, 0,
    0, 1, 3, 3, 2, 0, 7, 5, 4, 4, 6, 7,
    2, 3, 7, 7, 6, 2, 1, 0, 4, 4, 5, 1
  };

  vertex_buffer_->bind    ();
  vertex_buffer_->set_data(sizeof(float)    * vertices.size(), vertices.data());
  vertex_buffer_->unbind  ();

  index_buffer_ ->bind    ();
  index_buffer_ ->set_data(sizeof(unsigned) * indices .size(), indices .data());
  index_buffer_ ->unbind  ();

  draw_count_ = indices.size();
}
void volume_renderer::render    (const camera* camera)
{
  glm::ivec4 viewport;
  glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));

  exit_points_color_texture_->bind     ();
  exit_points_color_texture_->set_image(GL_RGBA, viewport[2], viewport[3], GL_RGBA, GL_FLOAT);
  exit_points_color_texture_->unbind   ();
  exit_points_depth_texture_->bind     ();
  exit_points_depth_texture_->set_image(GL_RED , viewport[2], viewport[3], GL_RED , GL_FLOAT);
  exit_points_depth_texture_->unbind   ();

  // Apply prepass.
  framebuffer_              ->bind();
  prepass_shader_program_   ->bind();
  prepass_vertex_array_     ->bind();
  index_buffer_             ->bind();

  prepass_shader_program_   ->set_uniform("projection", camera->projection_matrix      ());
  prepass_shader_program_   ->set_uniform("view"      , camera->inverse_absolute_matrix());

  glDrawElements(GL_TRIANGLES, draw_count_, GL_UNSIGNED_INT, nullptr);

  prepass_shader_program_   ->unbind();
  prepass_vertex_array_     ->unbind();
  index_buffer_             ->unbind();
  framebuffer_              ->unbind();

  // Apply main pass.
  shader_program_           ->bind();
  vertex_array_             ->bind();
  index_buffer_             ->bind();
                            
  shader_program_           ->set_uniform("projection", camera->projection_matrix      ());
  shader_program_           ->set_uniform("view"      , camera->inverse_absolute_matrix());
  
  gl::texture_2d::set_active(0);
  transfer_function_texture_->bind();
  gl::texture_2d::set_active(1);
  exit_points_color_texture_->bind();
  gl::texture_2d::set_active(2);
  volume_texture_           ->bind();

  shader_program_->set_uniform("step_size"        , 0.001F);
  shader_program_->set_uniform("screen_size"      , glm::uvec2(viewport[2], viewport[3]));
  shader_program_->set_uniform("transfer_function", 0);
  shader_program_->set_uniform("exit_points"      , 1);
  shader_program_->set_uniform("volume"           , 2);

  gl::texture_2d::set_active(2);
  volume_texture_           ->unbind();
  gl::texture_2d::set_active(1);
  exit_points_color_texture_->unbind();
  gl::texture_2d::set_active(0);
  transfer_function_texture_->unbind();

  glDrawElements(GL_TRIANGLES, draw_count_, GL_UNSIGNED_INT, nullptr);

  shader_program_        ->unbind();
  vertex_array_          ->unbind();
  index_buffer_          ->unbind();
}

void volume_renderer::set_data  (const uint3& dimensions, const float3& spacing, const float* retardation)
{
  // TODO: Transfer function, exit points, volume.
}
}
