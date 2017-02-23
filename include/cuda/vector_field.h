#ifndef PLI_VIS_VECTOR_FIELD_H_
#define PLI_VIS_VECTOR_FIELD_H_

#define _USE_MATH_DEFINES

#include <math.h>

#include <device_launch_parameters.h>
#include <vector_types.h>

#include <convert.h>
#include <vector_ops.h>

namespace pli
{
template<typename scalar_type, typename vector_type, typename color_type>
__global__ void create_vectors(
  const uint3        dimensions  ,
  const scalar_type* directions  ,
  const scalar_type* inclinations,
  const scalar_type  scale       ,
  const vector_type  spacing     ,
        vector_type* points      ,
        color_type*  colors      )
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x > dimensions.x ||
      y > dimensions.y ||
      z > dimensions.z)
    return;

  auto volume_index = z + dimensions.z * (y + dimensions.y * x);
  auto longitude    =          directions  [volume_index]  * M_PI / 180.0F;
  auto latitude     = (90.0F - inclinations[volume_index]) * M_PI / 180.0F;

  vector_type position = {
    x * spacing.x,
    y * spacing.y,
    z * spacing.z};

  auto minimum_spacing = min(spacing.x, min(spacing.y, spacing.z));
  auto vector_start    = cush::to_cartesian_coords(float3{ scale * minimum_spacing / 2.0, longitude, latitude});
  auto vector_end      = cush::to_cartesian_coords(float3{-scale * minimum_spacing / 2.0, longitude, latitude});
  auto unscaled        = cush::to_cartesian_coords(float3{1.0, longitude, latitude});
  auto color           = make_float4(abs(unscaled.x), abs(unscaled.y), abs(unscaled.z), 1.0);

  auto point_index = 2 * volume_index;
  points[point_index    ] = position + vector_start;
  points[point_index + 1] = position + vector_end  ;
  colors[point_index    ] = color;
  colors[point_index + 1] = color;
}

void create_vector_field(
  const uint3&  dimensions  ,
  const float*  directions  ,
  const float*  inclinations,
  const float&  scale       ,
  const float3& spacing     ,
        float3* points      ,
        float4* colors      );
}

#endif