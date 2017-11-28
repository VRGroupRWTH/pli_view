#ifndef PLI_VIS_POLAR_PLOT_FIELD_HPP_
#define PLI_VIS_POLAR_PLOT_FIELD_HPP_

#include <memory>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/opengl/all.hpp>

namespace pli
{
class polar_plot_field : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;

  void set_data  (const std::vector<float3>& vertices  , 
                  const std::vector<float3>& directions);

private:
  std::unique_ptr<gl::program>      shader_program_  ;
  std::unique_ptr<gl::vertex_array> vertex_array_    ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_   ;
  std::unique_ptr<gl::array_buffer> direction_buffer_;
  std::size_t                       draw_count_      = 0;
};
}

#endif