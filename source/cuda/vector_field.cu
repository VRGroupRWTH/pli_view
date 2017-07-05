#include /* implements */ <cuda/vector_field.h>

#include <chrono>
#include <iostream>

#include <thrust/device_vector.h>

#include <cuda/sh/launch.h>

namespace pli
{
void create_vector_field(
  const uint3&  dimensions  ,
  const float*  directions  ,
  const float*  inclinations,
  const float3& spacing     ,
  const float&  scale       ,
        float3* points      ,
        float4* colors      ,
  std::function<void(const std::string&)> status_callback)
{
  auto total_start = std::chrono::system_clock::now();

  status_callback("Allocating and copying directions and inclinations.");
  auto voxel_count = dimensions.x * dimensions.y * dimensions.z;
  thrust::device_vector<float> directions_vector  (voxel_count);
  thrust::device_vector<float> inclinations_vector(voxel_count);
  copy_n(directions  , voxel_count, directions_vector  .begin());
  copy_n(inclinations, voxel_count, inclinations_vector.begin());
  auto directions_ptr   = raw_pointer_cast(&directions_vector  [0]);
  auto inclinations_ptr = raw_pointer_cast(&inclinations_vector[0]);
  cudaDeviceSynchronize();
  
  status_callback("Creating vectors.");
  create_vector_field_internal<<<cush::grid_size_3d(dimensions), cush::block_size_3d()>>>(
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
  status_callback("Cuda operations took " + std::to_string(total_elapsed_seconds.count()) + " seconds.");
}

void create_vector_field(
  const uint3&  dimensions  , 
  const float3* unit_vectors, 
  const float3& spacing     , 
  const float&  scale       , 
  float3*       points      , 
  float4*       colors      , 
  std::function<void(const std::string&)> status_callback)
{
  auto total_start = std::chrono::system_clock::now();
  
  status_callback("Allocating and copying directions and inclinations.");
  auto voxel_count = dimensions.x * dimensions.y * dimensions.z;
  thrust::device_vector<float3> unit_vectors_vector(voxel_count);
  copy_n(unit_vectors, voxel_count, unit_vectors_vector.begin());
  auto unit_vectors_ptr = raw_pointer_cast(&unit_vectors_vector[0]);
  cudaDeviceSynchronize();
  
  status_callback("Creating vectors.");
  create_vector_field_internal<<<cush::grid_size_3d(dimensions), cush::block_size_3d()>>>(
    dimensions      , 
    unit_vectors_ptr,
    spacing         , 
    scale           , 
    points          ,
    colors          );
  cudaDeviceSynchronize();

  auto total_end = std::chrono::system_clock::now();
  std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
  status_callback("Cuda operations took " + std::to_string(total_elapsed_seconds.count()) + " seconds.");
}
}
