#include <pli_vis/visualization/algorithms/zernike_field.hpp>

#include <pli_vis/visualization/primitives/camera.hpp>
#include <shaders/simple_color_texture.vert.glsl>
#include <shaders/simple_color_texture.frag.glsl>
#include <shaders/zernike.vert.glsl>
#include <shaders/zernike.frag.glsl>

namespace pli
{
void zernike_field::initialize()
{
  std::vector<float>    vertices  = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  std::vector<float>    texcoords = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
  std::vector<unsigned> indices   = {0u, 1u, 2u, 0u, 2u, 3u};

  prepass_program_      = std::make_unique<gl::program>              ();
  prepass_vertex_array_ = std::make_unique<gl::vertex_array>         ();
  main_program_         = std::make_unique<gl::program>              ();
  main_vertex_array_    = std::make_unique<gl::vertex_array>         ();

  vertex_buffer_        = std::make_unique<gl::array_buffer>         ();
  texcoord_buffer_      = std::make_unique<gl::array_buffer>         ();
  index_buffer_         = std::make_unique<gl::index_buffer>         ();
  coefficient_buffer_   = std::make_unique<gl::shader_storage_buffer>();

  render_target_        = std::make_unique<render_target>            ();
  
  prepass_program_     ->attach_shader         (gl::vertex_shader  (shaders::zernike_vert));
  prepass_program_     ->attach_shader         (gl::fragment_shader(shaders::zernike_frag));
  prepass_program_     ->link                  ();
  prepass_vertex_array_->bind                  ();
  prepass_program_     ->bind                  ();
  vertex_buffer_       ->bind                  ();
  vertex_buffer_       ->set_data              (vertices.size() * sizeof(float), vertices.data());
  prepass_program_     ->set_attribute_buffer  ("position", 3, GL_FLOAT);
  prepass_program_     ->enable_attribute_array("position");
  vertex_buffer_       ->unbind                ();
  index_buffer_        ->bind                  ();
  index_buffer_        ->set_data              (indices.size() * sizeof(unsigned), indices.data());
  index_buffer_        ->unbind                ();
  prepass_vertex_array_->set_element_buffer    (*index_buffer_.get());
  prepass_program_     ->unbind                ();
  prepass_vertex_array_->unbind                ();
  
  main_program_        ->attach_shader         (gl::vertex_shader  (shaders::simple_color_texture_vert));
  main_program_        ->attach_shader         (gl::fragment_shader(shaders::simple_color_texture_frag));
  main_program_        ->link                  ();
  main_vertex_array_   ->bind                  ();
  main_program_        ->bind                  ();
  vertex_buffer_       ->bind                  ();
  vertex_buffer_       ->set_data              (vertices.size() * sizeof(float), vertices.data());
  main_program_        ->set_attribute_buffer  ("position", 3, GL_FLOAT);
  main_program_        ->enable_attribute_array("position");
  vertex_buffer_       ->unbind                ();
  texcoord_buffer_     ->bind                  ();
  texcoord_buffer_     ->set_data              (texcoords.size() * sizeof(float), texcoords.data());
  main_program_        ->set_attribute_buffer  ("texcoords", 2, GL_FLOAT);
  main_program_        ->enable_attribute_array("texcoords");
  texcoord_buffer_     ->unbind                ();
  index_buffer_        ->bind                  ();
  index_buffer_        ->set_data              (indices.size() * sizeof(unsigned), indices.data());
  index_buffer_        ->unbind                ();
  main_vertex_array_   ->set_element_buffer    (*index_buffer_.get());
  main_program_        ->unbind                ();
  main_vertex_array_   ->unbind                ();

  draw_count_ = indices.size();
} 
void zernike_field::render    (const camera* camera)
{
  const auto size = dimensions_ * spacing_;

  if(needs_update_)
  {
    render_target_       ->resize     (size);
    render_target_       ->bind       ();
    prepass_program_     ->bind       ();
    prepass_vertex_array_->bind       ();
    coefficient_buffer_  ->bind_base  (0);

    prepass_program_     ->set_uniform("dimensions"            , dimensions_            );
    prepass_program_     ->set_uniform("spacing"               , spacing_               );
    prepass_program_     ->set_uniform("coefficients_per_voxel", coefficients_per_voxel_);
    prepass_program_     ->set_uniform("color_mode"            , color_mode_            );
    prepass_program_     ->set_uniform("color_k"               , color_k_               );
    prepass_program_     ->set_uniform("color_inverted"        , color_inverted_        );
    glViewport             (0, 0, size.x, size.y);
    glClearColor           (0.0, 0.0, 0.0, 0.0);
    glClear                (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawElementsInstanced(GL_TRIANGLES, draw_count_, GL_UNSIGNED_INT, nullptr, primitive_count_);
  
    prepass_vertex_array_->unbind     ();
    prepass_program_     ->unbind     ();
    render_target_       ->unbind     ();

    needs_update_ = false;
  }
  
  gl::texture_2d::set_active(GL_TEXTURE0);
  
  render_target_    ->color_texture()->bind();
  main_program_     ->bind();
  main_vertex_array_->bind();
  
  main_program_     ->set_uniform("texture_unit", 0);
  main_program_     ->set_uniform("model"       , absolute_matrix                 ());
  main_program_     ->set_uniform("view"        , camera->inverse_absolute_matrix ());
  main_program_     ->set_uniform("projection"  , camera->projection_matrix       ());
  main_program_     ->set_uniform("size"        , glm::vec2(size));
  glDrawElements(GL_TRIANGLES, draw_count_, GL_UNSIGNED_INT, nullptr);
  
  main_vertex_array_->unbind();
  main_program_     ->unbind();
  render_target_    ->color_texture()->unbind();
}

void zernike_field::set_data  (const uint2& dimensions, const uint2& spacing, const unsigned coefficients_per_voxel, const std::vector<float>& coefficients)
{
  dimensions_             = {dimensions.x , dimensions.y};
  spacing_                = {spacing   .x , spacing   .y};
  coefficients_per_voxel_ = coefficients_per_voxel;
  primitive_count_        = dimensions_.x * dimensions_.y;

  coefficient_buffer_->bind     () ;
  coefficient_buffer_->set_data (coefficients.size() * sizeof(float), coefficients.data());
  coefficient_buffer_->bind_base(0);
  coefficient_buffer_->unbind   () ;
  
  needs_update_ = true;
}
}
