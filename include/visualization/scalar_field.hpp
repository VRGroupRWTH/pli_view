#ifndef PLI_VIS_SCALAR_FIELD_HPP_
#define PLI_VIS_SCALAR_FIELD_HPP_

#include <memory>

#include <vector_types.h>

#include <attributes/renderable.hpp>
#include <opengl/all.hpp>

namespace pli
{
class scalar_field : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;

  void set_data(
    const uint3&  dimensions,
    const float*  scalars   ,
    const float3& spacing   );

private:
  std::unique_ptr<gl::program>      shader_program_ ;
  std::unique_ptr<gl::vertex_array> vertex_array_   ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_  ;
  std::unique_ptr<gl::array_buffer> texcoord_buffer_;
  std::unique_ptr<gl::texture_2d>   texture_        ;
  std::size_t                       draw_count_     = 0;
};
}

#endif