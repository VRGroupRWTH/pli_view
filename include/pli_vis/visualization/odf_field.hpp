#ifndef PLI_VIS_ODF_FIELD_HPP_
#define PLI_VIS_ODF_FIELD_HPP_

#include <functional>
#include <memory>
#include <string>

#include <vector_types.h>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/opengl/all.hpp>

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
    const float3&  vector_spacing    ,
    const uint3&   vector_dimensions ,
    const float    scale             = 1.0  ,
    const bool     clustering        = false,
    const float    cluster_threshold = 0.0  ,
    std::function<void(const std::string&)> status_callback = [](const std::string&){});

  void set_visible_layers(
    const std::vector<bool>& visible_layers);

private:
  std::unique_ptr<gl::program>      shader_program_ ;
  std::unique_ptr<gl::vertex_array> vertex_array_   ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_  ;
  std::unique_ptr<gl::array_buffer> color_buffer_   ;
  std::unique_ptr<gl::index_buffer> index_buffer_   ;
  std::size_t                       draw_count_     = 0;
  uint3                             dimensions_     ;
  uint2                             tessellations_  ;
  std::vector<bool>                 visible_layers_ ;
};
}

#endif