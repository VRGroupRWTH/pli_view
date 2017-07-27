#ifndef PLI_VIS_SPHERICAL_HISTOGRAM_H_
#define PLI_VIS_SPHERICAL_HISTOGRAM_H_

#define _USE_MATH_DEFINES

#include <math.h>

#include <device_launch_parameters.h>
#include <vector_types.h>

namespace pli
{
// Call on a bin_dimensions.x x bin_dimensions.y 2D grid. 
// Vectors are in spherical coordinates.
template<typename vector_type>
__global__ void create_bins_kernel(
  uint2        bin_dimensions, 
  vector_type* bin_vectors   )
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= bin_dimensions.x || y >= bin_dimensions.y)
    return;

  auto& bin_vector = bin_vectors[x + bin_dimensions.x * y];
  bin_vector.x = 1.0;
  bin_vector.y = 2 * M_PI * x / bin_dimensions.x;

  if (x == 0 && y == 0)
    bin_vector.z = 0;
  else if (x == bin_dimensions.x - 1 && y == bin_dimensions.y - 1)
    bin_vector.z = M_PI;
  else
    bin_vector.z = M_PI * (y + 1) / (bin_dimensions.y + 1);
}

// Call on a vectors_size.x x vectors_size.y x vectors_size.z 3D grid.
// Vectors are in spherical coordinates.
template<typename vector_type, typename magnitude_type>
__global__ void accumulate_kernel(
  uint3                 vectors_size  ,
  uint3                 field_offset  ,
  uint3                 field_size    ,
  const vector_type*    vectors       ,
  uint2                 bin_dimensions,
  vector_type*          bin_vectors   ,
  magnitude_type*       bin_magnitudes)
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= vectors_size.x || y >= vectors_size.y || z >= vectors_size.z)
    return;

  auto  vector_index = z + field_size.z * (y + field_size.y * (x + field_offset.x) + field_offset.y) + field_offset.z;
  auto& vector       = vectors[vector_index];

  magnitude_type max_dot = -1; auto max_index = -1;
  magnitude_type min_dot =  1; auto min_index = -1;
  for (auto i = 0; i < bin_dimensions.x * bin_dimensions.y; i++)
  {
    auto& bin_vector = bin_vectors[i];

    magnitude_type dot = 
      cos(vector.z) * cos(bin_vector.z) + 
      sin(vector.z) * sin(bin_vector.z) * 
      cos(vector.y  - bin_vector.y);

    if (dot > max_dot) { max_dot = dot; max_index = i; }
    if (dot < min_dot) { min_dot = dot; min_index = i; }
  }
  if (max_index != -1) atomicAdd(&bin_magnitudes[max_index], 1.0);
  if (min_index != -1) atomicAdd(&bin_magnitudes[min_index], 1.0);
}
}

#endif
