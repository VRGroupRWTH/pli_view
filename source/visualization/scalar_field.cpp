#include /* implements */ <visualization/scalar_field.hpp>

#include <math/camera.hpp>
#include <shaders/scalar_field.vert.glsl>
#include <shaders/scalar_field.frag.glsl>

namespace pli
{
void scalar_field::initialize()
{
  shader_program_ .reset(new gl::program     );
  vertex_array_   .reset(new gl::vertex_array);
  vertex_buffer_  .reset(new gl::array_buffer);
  texcoord_buffer_.reset(new gl::array_buffer);
  texture_        .reset(new gl::texture_2d  );

  shader_program_ ->attach_shader(gl::vertex_shader  (shaders::scalar_field_vert));
  shader_program_ ->attach_shader(gl::fragment_shader(shaders::scalar_field_frag));
  shader_program_ ->link();
  
  shader_program_ ->bind();
  vertex_array_   ->bind();

  vertex_buffer_  ->bind  ();
  shader_program_ ->set_attribute_buffer  ("vertex", 3, GL_FLOAT);
  shader_program_ ->enable_attribute_array("vertex");
  vertex_buffer_  ->unbind();
  
  texcoord_buffer_->bind  ();
  shader_program_ ->set_attribute_buffer  ("texcoords", 2, GL_FLOAT);
  shader_program_ ->enable_attribute_array("texcoords");
  texcoord_buffer_->unbind();

  texture_        ->set_active(0);
  texture_        ->bind      ();
  texture_        ->min_filter(GL_NEAREST_MIPMAP_NEAREST);
  texture_        ->mag_filter(GL_NEAREST);
  texture_        ->wrap_s    (GL_CLAMP_TO_EDGE);
  texture_        ->wrap_t    (GL_CLAMP_TO_EDGE);

  shader_program_ ->set_uniform("texture_unit", 0);

  texture_        ->unbind();
  vertex_array_   ->unbind();
  shader_program_ ->unbind();
}
void scalar_field::render    (const camera* camera)
{
  shader_program_->bind  ();
  vertex_array_  ->bind  ();
  texture_       ->bind  ();

  shader_program_->set_uniform("projection", camera->projection_matrix      ());
  shader_program_->set_uniform("view"      , camera->inverse_absolute_matrix());

  glDrawArrays(GL_TRIANGLES, 0, draw_count_);

  texture_       ->unbind();
  vertex_array_  ->unbind();
  shader_program_->unbind();
}

void scalar_field::set_data(
  const uint3&  dimensions  ,
  const float*  scalars     ,
  const float3& spacing     )
{
  draw_count_ = 6 * dimensions.z;

  float3 size         = {spacing.x * dimensions.x, spacing.y * dimensions.y, spacing.z * dimensions.z};
  float3 vertices [6] = {{0,0,0}, {size.x,0,0}, {size.x,-size.y,0}, {0,0,0}, {size.x,-size.y,0}, {0,-size.y,0}};
  float2 texcoords[6] = {{0,0}, {0,1}, {1,1}, {0,0}, {1,1}, {1,0}};

  vertex_buffer_  ->bind    ();
  vertex_buffer_  ->set_data(draw_count_ * sizeof(float3), vertices);
  vertex_buffer_  ->unbind  ();
  
  texcoord_buffer_->bind    ();
  texcoord_buffer_->set_data(draw_count_ * sizeof(float2), texcoords);
  texcoord_buffer_->unbind  ();
  
  texture_->set_active(0);
  texture_->bind      ();
  texture_->set_image (GL_RED, dimensions.y, dimensions.x, GL_RED, GL_FLOAT, scalars);
  texture_->generate_mipmaps();
  texture_->unbind    ();
}
}

