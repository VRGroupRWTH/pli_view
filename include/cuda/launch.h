#ifndef PLI_VIS_LAUNCH_H_
#define PLI_VIS_LAUNCH_H_

#include <math.h>

#include <device_launch_parameters.h>

namespace pli
{
inline unsigned block_size_1d()
{
  return 64;
}
inline dim3     block_size_2d()
{
  return {32, 32, 1};
}
inline dim3     block_size_3d()
{
  return {16, 16, 4};
}

inline unsigned grid_size_1d(unsigned target_dimension )
{
  return ceil(float(target_dimension) / block_size_1d());
}
inline dim3     grid_size_2d(dim3     target_dimensions)
{
  auto block_size = block_size_2d();
  return {
    ceil(float(target_dimensions.x) / block_size.x),
    ceil(float(target_dimensions.y) / block_size.y),
    1 
  };
}
inline dim3     grid_size_3d(dim3     target_dimensions)
{
  auto block_size = block_size_3d();
  return{
    ceil(float(target_dimensions.x) / block_size.x),
    ceil(float(target_dimensions.y) / block_size.y),
    ceil(float(target_dimensions.z) / block_size.z)
  };
}
}

#endif
