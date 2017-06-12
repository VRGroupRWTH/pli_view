#ifndef PLI_ODF_SPHERICAL_HISTOGRAM_H_
#define PLI_ODF_SPHERICAL_HISTOGRAM_H_

#include <device_launch_parameters.h>
#include <vector_types.h>

namespace pli
{
// Call on a bin_dimensions.x x bin_dimensions.y 2D grid.
template<typename vector_type>
__global__ void create_bins(
  uint2        bin_dimensions, 
  vector_type* bin_vectors   )
{
  auto longitude_index = blockIdx.x * blockDim.x + threadIdx.x;
  auto latitude_index  = blockIdx.y * blockDim.y + threadIdx.y;

  if (longitude_index >= bin_dimensions.x ||
      latitude_index  >= bin_dimensions.y )
    return;

  auto& bin_vector = bin_vectors[longitude_index + bin_dimensions.x * latitude_index];
  bin_vector.x = 1.0;
  bin_vector.y = 2 * M_PI *  longitude_index     /  bin_dimensions.x;

  if (longitude_index == 0 && latitude_index == 0)
    bin_vector.z = 0;
  else if (longitude_index == bin_dimensions.x - 1 && latitude_index == bin_dimensions.y - 1)
    bin_vector.z = M_PI;
  else
    bin_vector.z = M_PI * (latitude_index + 1) / (bin_dimensions.y + 1);
}

// Call on a vectors_size.x x vectors_size.y x vectors_size.z 3D grid.
template<typename vector_type, typename magnitude_type>
__global__ void accumulate(
  uint3                 vectors_size  ,
  uint3                 field_offset  ,
  uint3                 field_size    ,
  const magnitude_type* longitudes    ,
  const magnitude_type* latitudes     ,
  uint2                 bin_dimensions,
  vector_type*          bin_vectors   ,
  magnitude_type*       bin_magnitudes)
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= vectors_size.x || y >= vectors_size.y || z >= vectors_size.z)
    return;

  auto  vector_index   = z + field_size.z * (y + field_size.y * (x + field_offset.x) + field_offset.y) + field_offset.z;
  auto& direction      = longitudes[vector_index];
  auto& inclination    = latitudes [vector_index];
  
  magnitude_type max_dot = -1; auto max_index = -1;
  magnitude_type min_dot =  1; auto min_index = -1;
  for (auto i = 0; i < bin_dimensions.x * bin_dimensions.y; i++)
  {
    auto& bin_vector = bin_vectors[i];

    magnitude_type dot = 
      cos(inclination) * cos(bin_vector.z) + 
      sin(inclination) * sin(bin_vector.z) * 
      cos(direction - bin_vector.y);

    if (dot > max_dot)
    {
      max_dot   = dot;
      max_index = i;
    }
    if (dot < min_dot)
    {
      min_dot   = dot;
      min_index = i;
    }
  }
  if (max_index != -1)
    atomicAdd(&bin_magnitudes[max_index], 1.0);
  if (min_index != -1)
    atomicAdd(&bin_magnitudes[min_index], 1.0);
}

  
// Call on a vectors_size.x x vectors_size.y x vectors_size.z 3D grid.
template<typename vector_type, typename magnitude_type>
__global__ void accumulate(
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
      cos(vector.y - bin_vector.y);

    if (dot > max_dot)
    {
      max_dot   = dot;
      max_index = i;
    }
    if (dot < min_dot)
    {
      min_dot   = dot;
      min_index = i;
    }
  }
  if (max_index != -1)
    atomicAdd(&bin_magnitudes[max_index], 1.0);
  if (min_index != -1)
    atomicAdd(&bin_magnitudes[min_index], 1.0);
}
}

#endif
