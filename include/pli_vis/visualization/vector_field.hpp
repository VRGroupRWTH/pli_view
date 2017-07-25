#ifndef PLI_VIS_VECTOR_FIELD_HPP_
#define PLI_VIS_VECTOR_FIELD_HPP_

#include <functional>
#include <memory>
#include <string>

#include <vector_types.h>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/opengl/all.hpp>

namespace pli
{
class vector_field : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;

  void set_data(
    const uint3&  dimensions  ,
    const float3* unit_vectors,
    float         scale       ,
    std::function<void(const std::string&)> status_callback = [](const std::string&){});

private:
  std::unique_ptr<gl::program>      shader_program_;
  std::unique_ptr<gl::vertex_array> vertex_array_  ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_ ;
  std::unique_ptr<gl::array_buffer> color_buffer_  ;
  std::size_t                       draw_count_    = 0;
};
}

#endif