#ifndef PLI_VIS_SAMPLE_H_
#define PLI_VIS_SAMPLE_H_

#include <math.h>

#include <device_launch_parameters.h>
#include <vector_types.h>

#include <vector_ops.h>

namespace pli
{
// Called on a branch voxels count 1D grid.
template<typename precision>
__global__ void create_branch_coefficients(
  const uint3    dimensions       ,
  const unsigned coefficient_count,
  precision*     coefficients     )
{
  auto voxel_count          = dimensions.x * dimensions.y * dimensions.z;
  auto coefficients_offset  = voxel_count * coefficient_count;

  auto tree_dimension_count = dimensions.z > 1 ? 3 : 2;
  auto tree_depth           = unsigned(logf(dimensions.x) / logf(2));
  //auto tree_voxel_count     = unsigned((pow(2, tree_dimension_count * (tree_depth + 1.0)) - 1.0) / (powf(2, tree_dimension_count) - 1.0));

  //auto index                = coefficients_offset + blockIdx.x * blockDim.x + threadIdx.x;
  
  //auto x                    = index / (dimensions.z * dimensions.y);
  //auto y                    = index /  dimensions.z % dimensions.y;
  //auto z                    = index %  dimensions.z;

  // Convert index to depth via leaf coefficient dimensions.
  // Get the depth of the tree from the index.
  // 0 1 2 3 - 4 5 6 7 - 8 9 10 11 - 12 13 14 15    16 17 18 19    20
  // 4*4*1 2*2*1 1*1*1
  // for(auto i = 0; i < 9; i++)
  //auto half_dimensions = 0.5 * dimensions;
  //for (auto i = 0; i < coefficient_count; i++)
  //  coefficients[31 + i] = coefficients[0] + coefficients[1] + coefficients[2] + coefficients[3];
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