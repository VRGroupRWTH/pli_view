#include /* implements */ <cuda/odf_field.h>

#include <chrono>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <cush.h>
#include <vector_ops.h>

namespace pli
{
void create_odfs(
  const uint3&    dimensions       ,
  const unsigned  coefficient_count,
  const float*    coefficients     ,
  const uint2&    tessellations    ,
  const float3&   spacing          ,
  const uint3&    block_size       ,
  const float     scale            ,
        float3*   points           ,
        float4*   colors           ,
        unsigned* indices          )
{
  auto total_start = std::chrono::system_clock::now();
  
  auto base_voxel_count   = dimensions.x * dimensions.y * dimensions.z;

  auto dimension_count    = dimensions.z > 1 ? 3 : 2;
  auto minimum_dimension  = min(dimensions.x, dimensions.y);
  if (dimension_count == 3)
    minimum_dimension = min(minimum_dimension, dimensions.z);

  auto max_depth          = log(minimum_dimension) / log(2);
  auto voxel_count        = unsigned(base_voxel_count * 
    ((1.0 - pow(1.0 / pow(2, dimension_count), max_depth + 1)) / 
     (1.0 -     1.0 / pow(2, dimension_count))));

  auto tessellation_count = tessellations.x * tessellations.y;
  auto point_count        = voxel_count * tessellation_count;

  std::cout << "Allocating and copying the leaf spherical harmonics coefficients." << std::endl;
  thrust::device_vector<float> coefficient_vectors(voxel_count * coefficient_count);
  copy_n(coefficients, base_voxel_count * coefficient_count, coefficient_vectors.begin());
  auto coefficients_ptr = raw_pointer_cast(&coefficient_vectors[0]);

  auto depth_offset     = 0;
  auto depth_dimensions = dimensions;
  for (auto depth = max_depth; depth >= 0; depth--)
  {
    if (depth != max_depth)
    {
      std::cout << "Calculating the depth " << depth << " coefficients." << std::endl;
      create_branch<<<dim3(depth_dimensions), 1>>>(
        dimensions       ,
        depth_dimensions ,
        depth_offset     ,
        coefficient_count,
        coefficients_ptr );
      cudaDeviceSynchronize();
    }

    std::cout << "Sampling sums of the coefficients." << std::endl;
    cush::sample_sums<<<dim3(depth_dimensions), 1>>>(
      depth_dimensions ,
      coefficient_count,
      tessellations    ,
      coefficients_ptr + depth_offset * coefficient_count , 
      points           + depth_offset * tessellation_count, 
      indices          + depth_offset * tessellation_count * 6,
      depth_offset     * tessellation_count);
    cudaDeviceSynchronize();
    
    depth_offset += depth_dimensions.x * depth_dimensions.y * depth_dimensions.z;
    depth_dimensions = {
      depth_dimensions.x / 2,
      depth_dimensions.y / 2,
      dimension_count == 3 ? depth_dimensions.z / 2 : 1
    };
  }

  std::cout << "Converting the points to Cartesian coordinates." << std::endl;
  thrust::transform(
    thrust::device,
    points,
    points + point_count,
    points,
    [] COMMON (const float3& point)
    {
      return cush::to_cartesian_coords(point);
    });
  cudaDeviceSynchronize();
  
  std::cout << "Normalizing the points." << std::endl;
  for (auto i = 0; i < voxel_count; i++)
  {
    float3* max_sample = thrust::max_element(
      thrust::device,
      points +  i      * tessellation_count,
      points + (i + 1) * tessellation_count,
      [] COMMON (const float3& lhs, const float3& rhs)
      {
        return length(lhs) < length(rhs);
      });
  
    thrust::transform(
      thrust::device,
      points +  i      * tessellation_count,
      points + (i + 1) * tessellation_count,
      points +  i      * tessellation_count,
      [max_sample] COMMON(float3 point)
      {
        auto max_sample_length = length(*max_sample);
        point.x /= max_sample_length;
        point.y /= max_sample_length;
        point.z /= max_sample_length;
        return point;
      });
  }
  cudaDeviceSynchronize();

  std::cout << "Assigning colors." << std::endl;
  thrust::transform(
    thrust::device,
    points,
    points + point_count,
    colors,
    [] COMMON (const float3& point)
    {
      // return make_float4(abs(point.x), abs(point.y), abs(point.z), 1.0); // Default
      return make_float4(abs(point.x), abs(point.z), abs(point.y), 1.0); // DMRI
    });
  cudaDeviceSynchronize();

  std::cout << "Translating and scaling the points." << std::endl;
  depth_offset     = 0;
  depth_dimensions = dimensions;
  for (auto depth = max_depth; depth >= 0; depth--)
  {
    auto depth_point_offset = 
      depth_offset * 
      tessellation_count;
    auto depth_point_count  =  
      depth_dimensions.x * 
      depth_dimensions.y * 
      depth_dimensions.z * 
      tessellation_count;
    
    uint3 depth_block_size {
      block_size.x * pow(2, max_depth - depth),
      block_size.y * pow(2, max_depth - depth),
      block_size.z * pow(2, max_depth - depth)};
    float3 offset{
      spacing.x * (depth_block_size.x - 1) * 0.5,
      spacing.y * (depth_block_size.y - 1) * 0.5,
      dimension_count == 3 ? spacing.z * (depth_block_size.z - 1) * 0.5 : 0.0};
    float3 depth_spacing {
      spacing.x * depth_block_size.x,
      spacing.y * depth_block_size.y,
      dimension_count == 3 ? spacing.z * depth_block_size.z : 1.0};
    auto depth_scale = scale * min(min(depth_spacing.x, depth_spacing.y), depth_spacing.z) * 0.5;

    thrust::transform(
      thrust::device,
      points + depth_point_offset,
      points + depth_point_offset + depth_point_count,
      points + depth_point_offset,
      [=] COMMON (const float3& point)
      {
        auto output = depth_scale * point;
        auto index  = int((&point - (points + depth_point_offset)) / tessellation_count);
        output.x += offset.x + depth_spacing.x * (index / (depth_dimensions.z * depth_dimensions.y));
        output.y += offset.y + depth_spacing.y * (index /  depth_dimensions.z % depth_dimensions.y);
        output.z += offset.z + depth_spacing.z * (index % depth_dimensions.z);
        return output;
      });
    cudaDeviceSynchronize();

    depth_offset += depth_dimensions.x * depth_dimensions.y * depth_dimensions.z;
    depth_dimensions = {
      depth_dimensions.x / 2,
      depth_dimensions.y / 2,
      dimension_count == 3 ? depth_dimensions.z / 2 : 1
    };
  }

  auto total_end = std::chrono::system_clock::now();
  std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
  std::cout << "Total elapsed time: " << total_elapsed_seconds.count() << "s." << std::endl;
}
}
