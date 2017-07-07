#ifndef PLI_VIS_BASIC_TRACER_HPP_
#define PLI_VIS_BASIC_TRACER_HPP_

#include <memory>

#include <boost/multi_array.hpp>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/opengl/all.hpp>

namespace pli
{
class streamline_tracer : public renderable
{
public:
  void set_data(const std::vector<float3>& points, const std::vector<float4>& colors);

  void initialize()                     override;
  void render    (const camera* camera) override;

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