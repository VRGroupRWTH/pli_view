#ifndef PLI_VIS_ZERNIKE_FIELD_HPP_
#define PLI_VIS_ZERNIKE_FIELD_HPP_

#include <memory>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/opengl/all.hpp>
#include <pli_vis/opengl/auxiliary/glm_uniforms.hpp>
#include <pli_vis/visualization/utility/render_target.hpp>

namespace pli
{
class zernike_field final : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;

  void set_data  (const uint2& dimensions, const uint2& spacing, const unsigned coefficients_per_voxel, const std::vector<float>& coefficients);

private:
  bool                                       needs_update_           = false;
  glm::uvec2                                 dimensions_             ;
  glm::uvec2                                 spacing_                ;
  unsigned                                   coefficients_per_voxel_ = 0;
  std::size_t                                draw_count_             = 0;
  std::size_t                                primitive_count_        = 0;

  std::unique_ptr<gl::program>               prepass_program_        ;
  std::unique_ptr<gl::vertex_array>          prepass_vertex_array_   ;
  std::unique_ptr<gl::program>               main_program_           ;
  std::unique_ptr<gl::vertex_array>          main_vertex_array_      ;

  std::unique_ptr<gl::array_buffer>          vertex_buffer_          ;
  std::unique_ptr<gl::array_buffer>          texcoord_buffer_        ;
  std::unique_ptr<gl::index_buffer>          index_buffer_           ;
  std::unique_ptr<gl::shader_storage_buffer> coefficient_buffer_     ;

  std::unique_ptr<render_target>             render_target_          ;
};
}

#endif