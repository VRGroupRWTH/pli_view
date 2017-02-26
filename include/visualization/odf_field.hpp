#ifndef PLI_VIS_ODF_FIELD_HPP_
#define PLI_VIS_ODF_FIELD_HPP_

#include <memory>

#include <vector_types.h>

#include <all.hpp>

#include <attributes/renderable.hpp>

namespace pli
{
class odf_field : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;

  void set_data(
    const uint3&   dimensions        ,
    const unsigned coefficient_count ,
    const float*   coefficients      ,
    const uint2&   tessellations     ,
    const float3&  spacing           ,
    const uint3&   block_size        ,
    const float    scale             = 1.0  ,
    const bool     clustering        = false,
    const float    cluster_threshold = 0.0  );

private:
  std::unique_ptr<gl::program>      shader_program_;
  std::unique_ptr<gl::vertex_array> vertex_array_  ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_ ;
  std::unique_ptr<gl::array_buffer> color_buffer_  ;
  std::unique_ptr<gl::index_buffer> index_buffer_  ;
  std::size_t                       draw_count_    = 0;
};
}

#endif