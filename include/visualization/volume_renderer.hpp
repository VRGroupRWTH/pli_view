#ifndef PLI_VIS_VOLUME_RENDERER_HPP_
#define PLI_VIS_VOLUME_RENDERER_HPP_

#include <memory>

#include <aspects/renderable.hpp>
#include <opengl/all.hpp>

namespace pli
{
class volume_renderer : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;

  void set_data             (const uint3& dimensions, const float3& spacing, const float* data);
  void set_transfer_function(const std::vector<float4>& transfer_function);
  void set_step_size        (float step_size);

private:
  std::unique_ptr<gl::program>      prepass_shader_program_   ;
  std::unique_ptr<gl::program>      shader_program_           ;

  std::unique_ptr<gl::vertex_array> prepass_vertex_array_     ;
  std::unique_ptr<gl::vertex_array> vertex_array_             ;

  std::unique_ptr<gl::array_buffer> vertex_buffer_            ;
  std::unique_ptr<gl::array_buffer> color_buffer_             ;
  std::unique_ptr<gl::index_buffer> index_buffer_             ;

  std::unique_ptr<gl::texture_1d>   transfer_function_texture_;
  std::unique_ptr<gl::texture_3d>   volume_texture_           ;
  
  std::unique_ptr<gl::framebuffer>  framebuffer_              ;
  std::unique_ptr<gl::texture_2d>   exit_points_color_texture_;
  std::unique_ptr<gl::texture_2d>   exit_points_depth_texture_;

  std::size_t                       draw_count_               = 0;
};
}

#endif