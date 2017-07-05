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
  GLint default_framebuffer_id;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &default_framebuffer_id);
  gl::framebuffer default_framebuffer(default_framebuffer_id);

  glm::ivec4 viewport;
  glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));

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
  
  // Set buffers for a unit cube.
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

  // Set textures.
  transfer_function_texture_->bind      ();
  transfer_function_texture_->wrap_s    (GL_REPEAT );
  transfer_function_texture_->min_filter(GL_NEAREST);
  transfer_function_texture_->mag_filter(GL_NEAREST);
  transfer_function_texture_->set_image (GL_RGBA32F, 256, GL_RGBA, GL_FLOAT);
  transfer_function_texture_->unbind    ();

  exit_points_color_texture_->bind      ();
  exit_points_color_texture_->wrap_s    (GL_REPEAT);
  exit_points_color_texture_->wrap_t    (GL_REPEAT);
  exit_points_color_texture_->min_filter(GL_NEAREST);
  exit_points_color_texture_->mag_filter(GL_NEAREST);
  exit_points_color_texture_->set_image (GL_RGBA32F, viewport[2], viewport[3], GL_RGBA, GL_FLOAT);
  exit_points_color_texture_->unbind    ();

  exit_points_depth_texture_->bind      ();
  exit_points_depth_texture_->wrap_s    (GL_REPEAT);
  exit_points_depth_texture_->wrap_t    (GL_REPEAT);
  exit_points_depth_texture_->min_filter(GL_NEAREST);
  exit_points_depth_texture_->mag_filter(GL_NEAREST);
  exit_points_depth_texture_->set_image (GL_DEPTH_COMPONENT24, viewport[2], viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT);
  exit_points_depth_texture_->unbind    ();

  volume_texture_           ->bind      ();
  volume_texture_           ->wrap_s    (GL_REPEAT);
  volume_texture_           ->wrap_t    (GL_REPEAT);
  volume_texture_           ->wrap_r    (GL_REPEAT);
  volume_texture_           ->min_filter(GL_LINEAR);
  volume_texture_           ->mag_filter(GL_LINEAR);
  volume_texture_           ->set_image (GL_R32F, 1, 1, 1, GL_RED, GL_FLOAT);
  volume_texture_           ->unbind    ();
  
  // Setup the framebuffer.
  framebuffer_->bind           ();
  framebuffer_->set_texture    (GL_COLOR_ATTACHMENT0, *exit_points_color_texture_.get());
  framebuffer_->set_texture    (GL_DEPTH_ATTACHMENT , *exit_points_depth_texture_.get());
  assert(framebuffer_->is_valid() && framebuffer_->is_complete());
  default_framebuffer.bind();
}
void volume_renderer::render    (const camera* camera)
{
  GLint default_framebuffer_id;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &default_framebuffer_id);
  gl::framebuffer default_framebuffer(default_framebuffer_id);

  glm::ivec4 viewport;
  glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(&viewport));

  // Apply prepass.
  exit_points_color_texture_->bind     ();
  exit_points_color_texture_->set_image(GL_RGBA32F, viewport[2], viewport[3], GL_RGBA, GL_FLOAT);
  exit_points_color_texture_->unbind   ();
  exit_points_depth_texture_->bind     ();
  exit_points_depth_texture_->set_image(GL_DEPTH_COMPONENT24, viewport[2], viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT);
  exit_points_depth_texture_->unbind   ();
  
  framebuffer_              ->bind();
  prepass_shader_program_   ->bind();
  prepass_vertex_array_     ->bind();
  index_buffer_             ->bind();

  prepass_shader_program_   ->set_uniform("projection", camera->projection_matrix      ());
  prepass_shader_program_   ->set_uniform("view"      , camera->inverse_absolute_matrix());

  glViewport    (viewport[0], viewport[1], viewport[2], viewport[3]);
  glClear       (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable      (GL_CULL_FACE);
  glCullFace    (GL_FRONT);
  glDrawElements(GL_TRIANGLES, draw_count_, GL_UNSIGNED_INT, nullptr);
  glDisable     (GL_CULL_FACE);

  prepass_shader_program_   ->unbind();
  prepass_vertex_array_     ->unbind();
  index_buffer_             ->unbind();
  default_framebuffer       .bind();
  
  // Apply main pass.
  shader_program_           ->bind();
  vertex_array_             ->bind();
  index_buffer_             ->bind();
       
  gl::texture_1d::set_active(GL_TEXTURE0);
  transfer_function_texture_->bind();
  gl::texture_2d::set_active(GL_TEXTURE1);
  exit_points_color_texture_->bind();
  gl::texture_3d::set_active(GL_TEXTURE2);
  volume_texture_           ->bind();
                       
  shader_program_           ->set_uniform("projection"       , camera->projection_matrix      ());
  shader_program_           ->set_uniform("view"             , camera->inverse_absolute_matrix());
  shader_program_           ->set_uniform("screen_size"      , glm::uvec2(viewport[2], viewport[3]));
  shader_program_           ->set_uniform("transfer_function", 0);
  shader_program_           ->set_uniform("exit_points"      , 1);
  shader_program_           ->set_uniform("volume"           , 2);
  
  glEnable      (GL_CULL_FACE);
  glCullFace    (GL_BACK);
  glDrawElements(GL_TRIANGLES, draw_count_, GL_UNSIGNED_INT, nullptr);
  glDisable     (GL_CULL_FACE);
  
  gl::texture_3d::set_active(GL_TEXTURE2);
  volume_texture_           ->unbind();
  gl::texture_2d::set_active(GL_TEXTURE1);
  exit_points_color_texture_->unbind();
  gl::texture_1d::set_active(GL_TEXTURE0);
  transfer_function_texture_->unbind();
 
  shader_program_           ->unbind();
  vertex_array_             ->unbind();
  index_buffer_             ->unbind();
}

void volume_renderer::set_data             (const uint3& dimensions, const float3& spacing, const float* data)
{
  volume_texture_->bind     ();
  volume_texture_->set_image(GL_R32F, dimensions.x, dimensions.y, dimensions.z, GL_RED, GL_FLOAT, data);
  volume_texture_->unbind   ();
}
void volume_renderer::set_transfer_function(const std::vector<float4>& transfer_function)
{                                    
  transfer_function_texture_->bind     ();
  transfer_function_texture_->set_image(GL_RGBA32F, 256, GL_RGBA, GL_FLOAT, transfer_function.data());
  transfer_function_texture_->unbind   ();
}
void volume_renderer::set_step_size        (float step_size)
{
  shader_program_->bind       ();
  shader_program_->set_uniform("step_size", step_size);
  shader_program_->unbind     ();
}
}
