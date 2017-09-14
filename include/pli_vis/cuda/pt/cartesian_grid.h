#ifndef CUPT_CARTESIAN_GRID_H_
#define CUPT_CARTESIAN_GRID_H_

#include <cstddef>

#include <host_defines.h>
#include <vector_types.h>

#include "../utility/vector_ops.h"

namespace cupt 
{
template<class type = float3>
class cartesian_grid
{
public:
  __host__ __device__ bool        is_within        (const float3&     point      )                   const 
  {
    return 
      dimensions.x * spacing.x > point.x && point.x >= 0 &&
      dimensions.y * spacing.y > point.y && point.y >= 0 &&
      dimensions.z * spacing.z > point.z && point.z >= 0;
  }
  __host__ __device__ uint3       grid_coords      (const float3&     point      )                   const 
  {
    const auto coords = floorf(point / spacing);
    return uint3 { 
      static_cast<unsigned>(coords.x),
      static_cast<unsigned>(coords.y),
      static_cast<unsigned>(coords.z)};
  }
  __host__ __device__ uint3       grid_coords      (const std::size_t id         )                   const 
  {
    return uint3 {id % dimensions.x, id / dimensions.x % dimensions.y, id / dimensions.x / dimensions.y};
  }
  __host__ __device__ float3      parametric_coords(const float3&     point      )                   const 
  {
    const auto base_point = point_coords(grid_coords(point));
    return float3 { 
      (point.x - base_point.x) / spacing.x,
      (point.y - base_point.y) / spacing.y,
      (point.z - base_point.z) / spacing.z};
  }
  __host__ __device__ float3      point_coords     (const uint3&      grid_coords)                   const 
  {
    return float3 { 
      grid_coords.x * spacing.x,
      grid_coords.y * spacing.y,
      grid_coords.z * spacing.z};
  }
  __host__ __device__ std::size_t point_id         (const uint3&      grid_coords)                   const 
  {
    return grid_coords.x + dimensions.x * (grid_coords.y + dimensions.y * grid_coords.z);
  }
  __host__ __device__ void        cell_point_ids   (const uint3&      grid_coords, std::size_t* ids) const 
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

  uint3       dimensions;
  float3      spacing   ;
  std::size_t data_size ;
  type*       data      = nullptr;
};
}

#endif
