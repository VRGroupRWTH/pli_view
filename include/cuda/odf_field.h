#ifndef PLI_VIS_SAMPLE_H_
#define PLI_VIS_SAMPLE_H_

#include <functional>
#include <math.h>

#include <device_launch_parameters.h>
#include <vector_types.h>

#include <spherical_harmonics.h>
#include <vector_ops.h>

namespace pli
{
void create_odfs(
  const uint3&    dimensions        ,
  const unsigned  coefficient_count ,
  const float*    coefficients      ,
  const uint2&    tessellations     , 
  const float3&   vector_spacing    ,
  const uint3&    vector_dimensions ,
  const float     scale             ,
        float3*   points            ,
        float4*   colors            ,
        unsigned* indices           ,
        bool      clustering        = false,
        float     cluster_threshold = 0.0  ,
        std::function<void(const std::string&)> status_callback = [](const std::string&){});
  
// Called on a layer_dimensions.x x layer_dimensions.y x layer_dimensions.z 3D grid.
template<typename precision>
__global__ void create_layer(
  const uint3    layer_dimensions  ,
  const unsigned layer_offset      ,
  const unsigned coefficient_count ,
  precision*     coefficients      ,
  bool           is_2d             ,
  bool           clustering        ,
  float          cluster_threshold )
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x >= layer_dimensions.x ||
      y >= layer_dimensions.y ||
      z >= layer_dimensions.z)
    return;

  auto  dimension_count         = is_2d ? 2 : 3;
  uint3 lower_layer_dimensions  = {layer_dimensions.x * 2, layer_dimensions.y * 2, dimension_count == 3 ? layer_dimensions.z * 2 : 1};
  auto  lower_layer_voxel_count = lower_layer_dimensions.x * lower_layer_dimensions.y * lower_layer_dimensions.z ;
  auto  lower_layer_offset      = layer_offset - lower_layer_voxel_count;
  auto  offset                  = coefficient_count * (layer_offset + z + layer_dimensions.z * (y + layer_dimensions.y * x));

  // Locate the associated voxels in the lower layer and sum them into this voxel.
  for (auto i = 0; i < 2; i++)
    for (auto j = 0; j < 2; j++)
      for (auto k = 0; k < dimension_count - 1; k++)
        for (auto c = 0; c < coefficient_count; c++)
          coefficients[offset + c] += coefficients[
            coefficient_count * 
              (lower_layer_offset + (2 * z + k) + lower_layer_dimensions.z * ((2 * y + j) + lower_layer_dimensions.y * (2 * x + i))) + c] 
            / powf(2, dimension_count);

  if (clustering)
  {
    // Compare this voxel to each associated voxel. 
    auto is_similar = true;
    for (auto i = 0; i < 2; i++)
      for (auto j = 0; j < 2; j++)
        for (auto k = 0; k < dimension_count - 1; k++)
        {
          auto other_offset = coefficient_count * (lower_layer_offset + (2 * z + k) + lower_layer_dimensions.z * ((2 * y + j) + lower_layer_dimensions.y * (2 * x + i)));
          if (cush::is_zero    (coefficient_count, coefficients + other_offset) ||
              cush::l2_distance(coefficient_count, coefficients + offset, coefficients + other_offset) > cluster_threshold)
            is_similar = false;
        }

    // If deemed similar, drop the associated voxels' coefficients.
    if (is_similar)
      for (auto i = 0; i < 2; i++)
        for (auto j = 0; j < 2; j++)
          for (auto k = 0; k < dimension_count - 1; k++)
          {
            auto other_offset = coefficient_count * (lower_layer_offset + (2 * z + k) + lower_layer_dimensions.z * ((2 * y + j) + lower_layer_dimensions.y * (2 * x + i)));
            for (auto c = 0; c < coefficient_count; c++)
              coefficients[other_offset + c] = 0.0;
          }
    // Else, drop this voxel's coefficients.
    else
      for (auto c = 0; c < coefficient_count; c++)
        coefficients[offset + c] = 0.0;
  }
}

}

#endif