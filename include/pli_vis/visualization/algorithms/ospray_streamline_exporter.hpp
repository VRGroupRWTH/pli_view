#ifndef PLI_VIS_OSPRAY_STREAMLINE_EXPORTER_HPP_
#define PLI_VIS_OSPRAY_STREAMLINE_EXPORTER_HPP_

#ifdef _WIN32
#define NOMINMAX
#endif

#include <string>
#include <vector>

#include <vector_types.h>

namespace pli
{
namespace ospray_streamline_exporter
{
void to_image(                 
  const float3&                position  , 
  const float3&                forward   , 
  const float3&                up        , 
  const uint2&                 size      ,
  const std::vector<float4>&   vertices  , 
  const std::vector<float4>&   tangents  ,
  const std::vector<unsigned>& gl_indices,
  const std::string&           filepath  );
};
}

#endif