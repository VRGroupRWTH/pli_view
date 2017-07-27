#ifndef PLI_VIS_BASIC_TRACER_HPP_
#define PLI_VIS_BASIC_TRACER_HPP_

#include <memory>
#include <vector>

#include <vector_types.h>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/opengl/all.hpp>

namespace pli
{
class streamline_renderer : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;
  
  void set_data(const std::vector<float3>& points, const std::vector<float3>& directions);
  void set_view_dependent_transparency (bool  enabled);
  void set_view_dependent_rate_of_decay(float value  );

private:
  std::unique_ptr<gl::program>      shader_program_  ;
  std::unique_ptr<gl::vertex_array> vertex_array_    ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_   ;
  std::unique_ptr<gl::array_buffer> direction_buffer_;
  std::size_t                       draw_count_                   = 0;
  bool                              view_dependent_transparency_  = true;
  float                             view_dependent_rate_of_decay_ = 1.0F;
};
}

#endif