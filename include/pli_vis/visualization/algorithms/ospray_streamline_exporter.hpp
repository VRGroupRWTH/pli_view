#ifndef PLI_VIS_OSPRAY_STREAMLINE_EXPORTER_HPP_
#define PLI_VIS_OSPRAY_STREAMLINE_EXPORTER_HPP_

#ifdef _WIN32
#define NOMINMAX
#endif

#include <string>
#include <vector>

#include <ospray/ospray_cpp.h>
#include <vector_types.h>

namespace pli
{
class ospray_streamline_exporter
{
public:
  void set_data(
    const std::vector<float4>& vertices, 
    const std::vector<float4>& tangents);
  void set_camera(
    const float3&              position, 
    const float3&              forward , 
    const float3&              up      = {0.0F, 1.0F, 0.0F});
  void set_image_size(
    const uint2&               size    );
  void save(                   
    const std::string&         filepath);

protected:
  std::vector<float4> vertices_       ;
  std::vector<float4> tangents_       ;
  float3              camera_position_;
  float3              camera_forward_ ;
  float3              camera_up_      ;
  uint2               image_size_     ;
};
}

#endif