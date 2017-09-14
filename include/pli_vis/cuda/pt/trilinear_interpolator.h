#ifndef CUPT_TRILINEAR_INTERPOLATOR_H_
#define CUPT_TRILINEAR_INTERPOLATOR_H_

#include <cstddef>

#include <host_defines.h>
#include <vector_types.h>

#include "../utility/vector_ops.h"
#include "cartesian_grid.h"

namespace cupt 
{
template<class type = float3>
class trilinear_interpolator 
{
public:
  __host__ __device__ explicit trilinear_interpolator(const cartesian_grid<type>* grid) : grid_(grid) { }
  __host__ __device__ bool     is_valid              (const float3& point) const
  {
    return grid_->is_within(point);
  }
  __host__ __device__ float3   interpolate           (const float3& point) const 
  {
    std::size_t cell_point_ids[8];
    grid_->cell_point_ids(grid_->grid_coords(point), cell_point_ids);

    float3 vectors[8];
    for (auto i = 0; i < 8; ++i)
      vectors[i] = grid_[cell_point_ids[i]];

    const float3 weights = grid_->parametric_coords(point);

    fold_along_axis<4>(vectors, weights.x);
    fold_along_axis<2>(vectors, weights.y);
    fold_along_axis<1>(vectors, weights.z);

    return vectors[0];
  }

private:
  template <std::size_t dims>
  __host__ __device__ void     fold_along_axis       (float3* values, const float weight) const 
  {
    for (auto i = 0; i < dims; ++i)
      values[i] = lerp(values[2 * i], values[2 * i + 1], weight);
  }

  const cartesian_grid<type>* grid_;
};
}

#endif