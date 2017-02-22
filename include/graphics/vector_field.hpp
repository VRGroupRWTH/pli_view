#ifndef PLI_VIS_VECTOR_FIELD_HPP_
#define PLI_VIS_VECTOR_FIELD_HPP_

#define GLP_CUDA_SUPPORT

#include <memory>

#include <all.hpp>
#include <vector_types.h>

#include <attributes/renderable.hpp>

namespace pli
{
class vector_field : public renderable
{
public:
  void initialize() override;
  void render    () override;

  void set_data(
    const float*  directions  ,
    const float*  inclinations,
    const uint3&  dimensions  ,
    const float3& spacing     ,
    float         scale       = 1.0);

private:
  std::unique_ptr<gl::program>      shader_program_;
  std::unique_ptr<gl::vertex_array> vertex_array_  ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_ ;
  std::unique_ptr<gl::array_buffer> color_buffer_  ;
  std::size_t                       draw_count_    = 0;
};
}

#endif