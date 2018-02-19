#ifndef PLI_VIS_STREAMLINE_RENDERER_HPP_
#define PLI_VIS_STREAMLINE_RENDERER_HPP_

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
  
  void set_data(
    const std::vector<float4>&   points    , 
    const std::vector<float4>&   directions,
    const std::vector<unsigned>& indices   );

private:
  std::size_t                       draw_count_      = 0;
  std::unique_ptr<gl::program>      program_         ;
  std::unique_ptr<gl::vertex_array> vertex_array_    ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_   ;
  std::unique_ptr<gl::array_buffer> direction_buffer_;
  std::unique_ptr<gl::index_buffer> index_buffer_    ;
};
}

#endif