#ifndef PLI_VIS_BASIC_TRACER_HPP_
#define PLI_VIS_BASIC_TRACER_HPP_

#include <memory>
#include <vector>

#include <vector_types.h>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/opengl/all.hpp>
#include <pli_vis/visualization/utility/render_target.hpp>

namespace pli
{
class streamline_renderer : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;
  
  void set_data      (
    const std::vector<float3>& points        , 
    const std::vector<float3>& directions    , 
    const std::vector<float4>& random_vectors);
  void set_ao_samples(const std::size_t& ao_samples);

  glm::uvec2  screen_size() const;
  std::size_t ao_samples () const;

private:     
  void initialize_normal_depth_pass(const glm::uvec2& screen_size);
  void initialize_color_pass       (const glm::uvec2& screen_size);
  void initialize_zoom_pass        (const glm::uvec2& screen_size);
  void initialize_main_pass        (const glm::uvec2& screen_size);

  void render_normal_depth_pass    (const camera* camera, const glm::uvec2& screen_size) const;
  void render_color_pass           (const camera* camera, const glm::uvec2& screen_size) const;
  void render_zoom_pass            (const camera* camera, const glm::uvec2& screen_size) const;
  void render_main_pass            (const camera* camera, const glm::uvec2& screen_size) const;
  
  std::size_t                          draw_count_               = 0 ;
  std::size_t                          ao_samples_               = 32;
                                       
  // Common data.                      
  std::unique_ptr<gl::array_buffer>    vertex_buffer_            ;
  std::unique_ptr<gl::array_buffer>    direction_buffer_         ;
                                       
  // Normal depth pass data.           
  std::unique_ptr<render_target>       normal_depth_map_         ;
  std::unique_ptr<gl::program>         normal_depth_program_     ;
  std::unique_ptr<gl::vertex_array>    normal_depth_vertex_array_;
                                       
  // Color pass data.                  
  std::unique_ptr<render_target>       color_map_                ;
  std::unique_ptr<gl::program>         color_program_            ;
  std::unique_ptr<gl::vertex_array>    color_vertex_array_       ;
                                       
  // Zoom pass data.                   
  std::unique_ptr<render_target>       zoom_map_                 ;
  std::unique_ptr<gl::program>         zoom_program_             ;
  std::unique_ptr<gl::vertex_array>    zoom_vertex_array_        ;
                                       
  // Main pass data.                   
  std::unique_ptr<gl::program>         program_                  ;
  std::unique_ptr<gl::vertex_array>    vertex_array_             ;
  std::unique_ptr<gl::texture_3d>      random_texture_           ;
};
}

#endif