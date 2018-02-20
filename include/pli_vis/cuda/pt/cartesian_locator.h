#ifndef CUPT_CARTESIAN_LOCATOR_H_
#define CUPT_CARTESIAN_LOCATOR_H_

#include <cstddef>

#include <host_defines.h>
#include <vector_types.h>

#include "../utility/vector_ops.h"

namespace cupt 
{
template<class type, class index_type = uint3>
class cartesian_locator
{
public:
  __host__ __device__ bool        is_within        (const type&       point      )                   const
  {
    return 
      dimensions.x * spacing.x > point.x && point.x >= 0 &&
      dimensions.y * spacing.y > point.y && point.y >= 0 &&
      dimensions.z * spacing.z > point.z && point.z >= 0;
  }
  __host__ __device__ index_type  grid_coords      (const type&       point      )                   const
  {
    const auto coords = floorf(point / spacing);
    return index_type {
      static_cast<unsigned>(coords.x),
      static_cast<unsigned>(coords.y),
      static_cast<unsigned>(coords.z)};
  }
  __host__ __device__ index_type  grid_coords      (const std::size_t id         )                   const
  {
    return index_type {id % dimensions.x, id / dimensions.x % dimensions.y, id / dimensions.x / dimensions.y};
  }
  __host__ __device__ type        parametric_coords(const type&       point      )                   const
  {
    const auto base_point = point_coords(grid_coords(point));
    return type {
      (point.x - base_point.x) / spacing.x,
      (point.y - base_point.y) / spacing.y,
      (point.z - base_point.z) / spacing.z};
  }
  __host__ __device__ type        point_coords     (const index_type& grid_coords)                   const
  {
    return type {
      grid_coords.x * spacing.x,
      grid_coords.y * spacing.y,
      grid_coords.z * spacing.z};
  }
  __host__ __device__ std::size_t point_id         (const index_type& grid_coords)                   const
  {
    return grid_coords.x + dimensions.x * (grid_coords.y + dimensions.y * grid_coords.z);
  }
  __host__ __device__ void        cell_point_ids   (const index_type& grid_coords, std::size_t* ids) const
  {
    const auto id = point_id(grid_coords);
    ids[0] = id                     ;
    ids[1] = id + 1                 ; 
    ids[2] = id +  dimensions.x     ;
    ids[3] = id +  dimensions.x + 1 ; 
    ids[4] = id +  dimensions.x      * dimensions.y      ;
    ids[5] = id +  dimensions.x      * dimensions.y + 1  ;
    ids[6] = id + (dimensions.x + 1) * dimensions.y      ;
    ids[7] = id + (dimensions.x + 1) * dimensions.y + 1  ;
  }

  index_type dimensions;
  type       spacing   ;
};
}

#endif
