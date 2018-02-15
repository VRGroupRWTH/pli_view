#ifndef PLI_VIS_OSPRAY_STREAMLINE_RENDERER_HPP_
#define PLI_VIS_OSPRAY_STREAMLINE_RENDERER_HPP_

#ifdef _WIN32
#define NOMINMAX
#endif

#include <memory>

#include <ospray/ospray_cpp.h>
#include <vector_types.h>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/opengl/all.hpp>

namespace pli
{
class ospray_streamline_renderer : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;
  
  void set_data(
    const std::vector<float3>& points    , 
    const std::vector<float3>& directions);
  
protected:
  std::size_t                               draw_count_     = 0;
  std::unique_ptr<gl::program>              program_        ;
  std::unique_ptr<gl::vertex_array>         vertex_array_   ;
  std::unique_ptr<gl::vertex_buffer>        vertex_buffer_  ;
  std::unique_ptr<gl::vertex_buffer>        texcoord_buffer_;
  std::unique_ptr<gl::index_buffer>         index_buffer_   ;
  std::unique_ptr<gl::texture_2d>           texture_        ;
  std::unique_ptr<ospray::cpp::Renderer>    renderer_       ;
  std::unique_ptr<ospray::cpp::Model>       model_          ;
  std::unique_ptr<ospray::cpp::Geometry>    streamlines_    ;
  std::unique_ptr<ospray::cpp::Camera>      camera_         ;
  std::unique_ptr<ospray::cpp::Data>        lights_         ;
  std::unique_ptr<ospray::cpp::FrameBuffer> framebuffer_    ;
};
}

#endif