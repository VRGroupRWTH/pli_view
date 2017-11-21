#include <pli_vis/visualization/algorithms/zernike_field.hpp>

#include <pli_vis/visualization/primitives/camera.hpp>
#include <shaders/zernike.vert.glsl>
#include <shaders/zernike.frag.glsl>

namespace pli
{
void zernike_field::initialize()
{
  std::vector<float>    vertices = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  std::vector<unsigned> indices  = {0u, 1u, 2u, 0u, 2u, 3u};

  program_            = std::make_unique<gl::program>              ();
  vertex_array_       = std::make_unique<gl::vertex_array>         ();
  vertex_buffer_      = std::make_unique<gl::array_buffer>         ();
  index_buffer_       = std::make_unique<gl::index_buffer>         ();
  coefficient_buffer_ = std::make_unique<gl::shader_storage_buffer>();
  
  program_      ->attach_shader         (gl::vertex_shader  (shaders::zernike_vert));
  program_      ->attach_shader         (gl::fragment_shader(shaders::zernike_frag));
  program_      ->link                  ();
  
  vertex_array_ ->bind                  ();
  program_      ->bind                  ();

  vertex_buffer_->bind                  ();
  vertex_buffer_->set_data              (vertices.size() * sizeof(float), vertices.data());
  program_      ->set_attribute_buffer  ("position", 3, GL_FLOAT);
  program_      ->enable_attribute_array("position");
  vertex_buffer_->unbind                ();

  index_buffer_ ->bind                  ();
  index_buffer_ ->set_data              (indices.size() * sizeof(unsigned), indices.data());
  index_buffer_ ->unbind                ();
  vertex_array_ ->set_element_buffer    (*index_buffer_.get());

  program_      ->unbind                ();
  vertex_array_ ->unbind                ();

  draw_count_ = 6;
}
void zernike_field::render    (const camera* camera)
{
  program_           ->bind();
  vertex_array_      ->bind();
  coefficient_buffer_->bind_base(0);
  
  program_->set_uniform("model"                 , absolute_matrix                ());
  program_->set_uniform("view"                  , camera->inverse_absolute_matrix());
  program_->set_uniform("projection"            , camera->projection_matrix      ());
  program_->set_uniform("dimensions"            , dimensions_                      );
  program_->set_uniform("spacing"               , spacing_                         );
  program_->set_uniform("coefficients_per_voxel", coefficients_per_voxel_          );
  glDrawElementsInstanced(GL_TRIANGLES, draw_count_, GL_UNSIGNED_INT, nullptr, primitive_count_);
  
  vertex_array_      ->unbind();
  program_           ->unbind();
}

void zernike_field::set_data(const uint2& dimensions, const uint2& spacing, const unsigned coefficients_per_voxel, const std::vector<float>& coefficients)
{
  dimensions_             = {dimensions.x , dimensions.y};
  spacing_                = {spacing   .x , spacing   .y};
  coefficients_per_voxel_ = coefficients_per_voxel;
  primitive_count_        = dimensions_.x * dimensions_.y;

  coefficient_buffer_->bind     () ;
  coefficient_buffer_->set_data (coefficients.size() * sizeof(float), coefficients.data());
  coefficient_buffer_->bind_base(0);
  coefficient_buffer_->unbind   () ;
}
}
