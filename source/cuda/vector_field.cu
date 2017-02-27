#include /* implements */ <cuda/vector_field.h>

#include <chrono>
#include <iostream>

#include <thrust/device_vector.h>

#include <cuda/launch.h>

namespace pli
{
void create_vector_field(
  const uint3&  dimensions  ,
  const float*  directions  ,
  const float*  inclinations,
  const float3& spacing     ,
  const float&  scale       ,
        float3* points      ,
        float4* colors      )
{
  auto total_start = std::chrono::system_clock::now();

  std::cout << "Allocating and copying directions and inclinations." << std::endl;
  auto voxel_count = dimensions.x * dimensions.y * dimensions.z;
  thrust::device_vector<float> directions_vector  (voxel_count);
  thrust::device_vector<float> inclinations_vector(voxel_count);
  copy_n(directions  , voxel_count, directions_vector  .begin());
  copy_n(inclinations, voxel_count, inclinations_vector.begin());
  auto directions_ptr   = raw_pointer_cast(&directions_vector  [0]);
  auto inclinations_ptr = raw_pointer_cast(&inclinations_vector[0]);
  cudaDeviceSynchronize();
  
  std::cout << "Creating vectors." << std::endl;
  create_vector_field_internal<<<grid_size_3d(dimensions), block_size_3d()>>>(
    dimensions      , 
    directions_ptr  , 
    inclinations_ptr,
    spacing         , 
    scale           , 
    points          ,
    colors          );
  cudaDeviceSynchronize();

  auto total_end = std::chrono::system_clock::now();
  std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
  std::cout << "Total elapsed time: " << total_elapsed_seconds.count() << "s." << std::endl;
}
}
