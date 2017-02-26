#ifndef PLI_VIS_SAMPLE_H_
#define PLI_VIS_SAMPLE_H_

#include <math.h>

#include <device_launch_parameters.h>
#include <vector_types.h>

#include <spherical_harmonics.h>
#include <vector_ops.h>

namespace pli
{
// Called on a depth_dimensions.x x depth_dimensions.y x depth_dimensions.z 3D grid.
template<typename precision>
__global__ void create_branch(
  const uint3    dimensions        ,
  const uint3    depth_dimensions  ,
  const unsigned depth_offset      ,
  const unsigned coefficient_count ,
  precision*     coefficients      ,
  bool           clustering        ,
  float          cluster_threshold )
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x > depth_dimensions.x ||
      y > depth_dimensions.y ||
      z > depth_dimensions.z)
    return;

  auto dimension_count = dimensions.z > 1 ? 3 : 2;

  uint3 lower_depth_dimensions  {
    depth_dimensions.x * 2,
    depth_dimensions.y * 2,
    dimension_count == 3 ? depth_dimensions.z * 2 : 1
  };

  auto lower_depth_voxel_count =
    lower_depth_dimensions.x *
    lower_depth_dimensions.y *
    lower_depth_dimensions.z ;
  
  auto lower_depth_offset =
    depth_offset - lower_depth_voxel_count;
  
  auto linear_index       = depth_offset + z + depth_dimensions.z * (y + depth_dimensions.y * x);
  auto coefficients_start = linear_index * coefficient_count;

  // Find the 2^dims voxels from the spatially lower layer and sum them to coefficients[coefficients_start].
  for (auto i = 0; i < 2; i++)
    for (auto j = 0; j < 2; j++)
      for (auto k = 0; k < (dimension_count == 3 ? 2 : 1); k++)
      {
        auto lower_start_index        = lower_depth_offset + (2 * z + k) + lower_depth_dimensions.z * ((2 * y + j) + lower_depth_dimensions.y * (2 * x + i));
        auto lower_coefficients_start = lower_start_index * coefficient_count;

        for (auto c = 0; c < coefficient_count; c++)
          coefficients[coefficients_start + c] += coefficients[lower_coefficients_start + c] / powf(2, dimension_count);
      }

  if (clustering)
  {
    // Compare the sum to its components. If it is deemed similar with each, drop the components' coefficients. Else, drop this voxel's coefficients.
    auto is_similar = true;
    for (auto i = 0; i < 2; i++)
      for (auto j = 0; j < 2; j++)
        for (auto k = 0; k < (dimension_count == 3 ? 2 : 1); k++)
        {
          auto lower_start_index        = lower_depth_offset + (2 * z + k) + lower_depth_dimensions.z * ((2 * y + j) + lower_depth_dimensions.y * (2 * x + i));
          auto lower_coefficients_start = lower_start_index * coefficient_count;

          auto difference = cush::compare(coefficient_count, coefficients + coefficients_start, coefficients + lower_coefficients_start);
          printf("Difference: %f \n", difference);
          if (difference > cluster_threshold)
            is_similar = false;
        }
    
    if (is_similar)
    {
      for (auto i = 0; i < 2; i++)
        for (auto j = 0; j < 2; j++)
          for (auto k = 0; k < (dimension_count == 3 ? 2 : 1); k++)
          {
            auto lower_start_index        = lower_depth_offset + (2 * z + k) + lower_depth_dimensions.z * ((2 * y + j) + lower_depth_dimensions.y * (2 * x + i));
            auto lower_coefficients_start = lower_start_index * coefficient_count;

            for (auto c = 0; c < coefficient_count; c++)
              coefficients[lower_coefficients_start + c] = 0.0;
          }
    }
    else
    {
      for (auto c = 0; c < coefficient_count; c++)
        coefficients[coefficients_start + c] = 0.0;
    }
  }
}

void create_odfs(
  const uint3&    dimensions        ,
  const unsigned  coefficient_count ,
  const float*    coefficients      ,
  const uint2&    tessellations     ,
  const float3&   spacing           ,
  const uint3&    block_size        ,
  const float     scale             ,
        float3*   points            ,
        float4*   colors            ,
        unsigned* indices           ,
        bool      clustering        = false,
        float     cluster_threshold = 0.0  );
}

#endif