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
    const uint3&   dimensions       ,
    const unsigned vectors_per_point,
    const float3*  unit_vectors     ,
    std::function<void(const std::string&)> status_callback = [](const std::string&){});
  void set_scale                       (float scale  );
  void set_view_dependent_transparency (bool  enabled);
  void set_view_dependent_rate_of_decay(float value  );

private:
  std::unique_ptr<gl::program>      shader_program_  ;
  std::unique_ptr<gl::vertex_array> vertex_array_    ;
  std::unique_ptr<gl::array_buffer> direction_buffer_;
  glm::uvec3                        dimensions_;
  std::size_t                       draw_count_                   = 0;
  unsigned                          vectors_per_point_            = 1;
  float                             scale_                        = 1.0F ;
  bool                              view_dependent_transparency_  = false;
  float                             view_dependent_rate_of_decay_ = 1.0F ;
};
}

#endif