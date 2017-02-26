#ifndef PLI_VIS_SAMPLE_H_
#define PLI_VIS_SAMPLE_H_

#include <math.h>

#include <device_launch_parameters.h>
#include <vector_types.h>

#include <vector_ops.h>

namespace pli
{
// Called on a depth_dimensions.x x depth_dimensions.y x depth_dimensions.z 3D grid.
template<typename precision>
__global__ void create_branch(
  const uint3    dimensions       ,
  const uint3    depth_dimensions ,
  const unsigned depth_offset     ,
  const unsigned coefficient_count,
  precision*     coefficients     )
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x > depth_dimensions.x ||
      y > depth_dimensions.y ||
      z > depth_dimensions.z)
    return;

  auto linear_index       = depth_offset + z + depth_dimensions.z * (y + depth_dimensions.y * x);
  auto coefficients_start = linear_index * coefficient_count;

  // TODO Find the 2^dims voxels from the spatially lower layer and sum them to coefficients[coefficients_index].
  for (auto i = 0; i < powf(2, dimensions.z > 1 ? 3 : 2); i++)
    for (auto c = 0; c < coefficient_count; c++)
      coefficients[coefficients_start + c] += coefficients[i * dimensions.x * dimensions.y * dimensions.z * coefficient_count + c];
}

void create_odfs(
  const uint3&    dimensions       ,
  const unsigned  coefficient_count,
  const float*    coefficients     ,
  const uint2&    tessellations    ,
  const float3&   spacing          ,
  const uint3&    block_size       ,
  const float     scale            ,
        float3*   points           ,
        float4*   colors           ,
        unsigned* indices          );
}

#endif