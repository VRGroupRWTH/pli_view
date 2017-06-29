#ifndef PLI_VIS_VOLUME_RENDERER_HPP_
#define PLI_VIS_VOLUME_RENDERER_HPP_

#include <memory>

#include <attributes/renderable.hpp>
#include <opengl/all.hpp>

namespace pli
{
class volume_renderer : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;

private:
  std::unique_ptr<gl::program>      shader_program_;
  std::unique_ptr<gl::vertex_array> vertex_array_  ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_ ;
  std::unique_ptr<gl::array_buffer> color_buffer_  ;
  std::size_t                       draw_count_    = 0;
};
}

#endif