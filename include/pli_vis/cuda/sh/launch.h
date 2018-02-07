#ifndef PLI_VIS_LAUNCH_H_
#define PLI_VIS_LAUNCH_H_

#include <vector_types.h>

namespace pli
{
__forceinline__ __host__ __device__ unsigned block_size_1d()
{
  return 64;
}
__forceinline__ __host__ __device__ dim3     block_size_2d()
{
  return {32, 32, 1};
}
__forceinline__ __host__ __device__ dim3     block_size_3d()
{
  return {16, 16, 4};
}
  
__forceinline__ __host__ __device__ unsigned grid_size_1d(unsigned target_dimension )
{
  auto block_size = block_size_1d();
  return unsigned((target_dimension + block_size - 1) / block_size);
}
__forceinline__ __host__ __device__ dim3     grid_size_2d(dim3     target_dimensions)
{
  auto block_size = block_size_2d();
  return {
    unsigned((target_dimensions.x + block_size.x - 1) / block_size.x),
    unsigned((target_dimensions.y + block_size.y - 1) / block_size.y),
    1u
  };
}
__forceinline__ __host__ __device__ dim3     grid_size_3d(dim3     target_dimensions)
{
  auto block_size = block_size_3d();
  return {
    unsigned((target_dimensions.x + block_size.x - 1) / block_size.x),
    unsigned((target_dimensions.y + block_size.y - 1) / block_size.y),
    unsigned((target_dimensions.z + block_size.z - 1) / block_size.z)
  };
}
}

#endif
