#ifndef PLI_VIS_LAUNCH_H_
#define PLI_VIS_LAUNCH_H_

#include <math.h>

#include <device_launch_parameters.h>

#include <pli_vis/cuda/sh/config.h>

namespace pli
{
INLINE COMMON unsigned block_size_1d()
{
  return 64;
}
INLINE COMMON dim3     block_size_2d()
{
  return {32, 32, 1};
}
INLINE COMMON dim3     block_size_3d()
{
  return {16, 16, 4};
}
  
INLINE COMMON unsigned grid_size_1d(unsigned target_dimension )
{
  auto block_size = block_size_1d();
  return unsigned((target_dimension + block_size - 1) / block_size);
}
INLINE COMMON dim3     grid_size_2d(dim3     target_dimensions)
{
  auto block_size = block_size_2d();
  return {
    unsigned((target_dimensions.x + block_size.x - 1) / block_size.x),
    unsigned((target_dimensions.y + block_size.y - 1) / block_size.y),
    1u
  };
}
INLINE COMMON dim3     grid_size_3d(dim3     target_dimensions)
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
