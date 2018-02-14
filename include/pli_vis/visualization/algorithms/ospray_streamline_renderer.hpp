#ifndef PLI_VIS_OSPRAY_STREAMLINE_RENDERER_HPP_
#define PLI_VIS_OSPRAY_STREAMLINE_RENDERER_HPP_

#include <vector_types.h>

#include <pli_vis/aspects/renderable.hpp>
#include <pli_vis/visualization/primitives/camera.hpp>

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

};
}

#endif