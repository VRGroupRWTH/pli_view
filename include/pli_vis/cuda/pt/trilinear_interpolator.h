#ifndef CUPT_TRILINEAR_INTERPOLATOR_H_
#define CUPT_TRILINEAR_INTERPOLATOR_H_

#include <cstddef>

#include <host_defines.h>
#include <vector_types.h>

#include "../utility/vector_ops.h"
#include "cartesian_locator.h"

namespace cupt 
{
template<class type>
class trilinear_interpolator 
{
public:
  __host__ __device__ explicit trilinear_interpolator(const uint3& dimensions, const type& spacing, const type* data) 
  : data_   {data}
  , locator_{dimensions, spacing}
  {
    
  }
  __host__ __device__ bool     is_valid              (const type& point) const
  {
    return locator_.is_within(point);
  }
  __host__ __device__ type     interpolate           (const type& point) const
  {
    std::size_t cell_point_ids[8];
    locator_.cell_point_ids(locator_.grid_coords(point), cell_point_ids);

    type vectors[8];
    for (auto i = 0; i < 8; ++i)
      vectors[i] = data_[cell_point_ids[i]];

    const type weights = locator_.parametric_coords(point);

    fold_along_axis<4>(vectors, weights.x);
    fold_along_axis<2>(vectors, weights.y);
    fold_along_axis<1>(vectors, weights.z);

    return vectors[0];
  }

private:
  template <std::size_t dimensions>
  __host__ __device__ void     fold_along_axis       (type* values, const float weight) const 
  {
    for (auto i = 0; i < dimensions; ++i)
      values[i] = lerp(values[2 * i], values[2 * i + 1], weight);
  }

  const type*             data_   ;
  cartesian_locator<type> locator_;
};
}

#endif