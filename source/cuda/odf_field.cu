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
  
  auto voxel_count            = dimensions.x * dimensions.y * dimensions.z;
  auto tessellation_count     = tessellations.x * tessellations.y;
  auto coefficients_size      = voxel_count * coefficient_count ;
  auto point_count            = voxel_count * tessellation_count;
  
  auto tree_dimension_count   = dimensions.z > 1 ? 3 : 2;
  auto tree_depth             = unsigned(log(dimensions.x) / log(2));
  auto tree_voxel_count       = unsigned((pow(2, tree_dimension_count * (tree_depth + 1.0)) - 1.0) / (pow(2, tree_dimension_count) - 1.0));
  auto tree_coefficients_size = tree_voxel_count * coefficient_count ;
  auto tree_point_count       = tree_voxel_count * tessellation_count;

  std::cout << "Allocating and copying the leaf spherical harmonics coefficients." << std::endl;
  thrust::device_vector<float> coefficient_vectors(tree_coefficients_size);
  copy_n(coefficients, coefficients_size, coefficient_vectors.begin());
  auto coefficients_ptr = raw_pointer_cast(&coefficient_vectors[0]);

  std::cout << "Calculating the branch coefficients." << std::endl;
  create_branch_coefficients<<<dim3(tree_voxel_count - voxel_count), 1>>>(
    dimensions       ,
    coefficient_count,
    coefficients_ptr );
  cudaDeviceSynchronize();

  std::cout << "Sampling sums of the coefficients." << std::endl;
  // TODO: FIX FOR TREE.
  cush::sample_sums<<<dim3(dimensions), 1>>>(
    dimensions       ,
    coefficient_count,
    tessellations    ,
    coefficients_ptr , 
    points           , 
    indices          );
  cudaDeviceSynchronize();

  std::cout << "Converting the points to Cartesian coordinates." << std::endl;
  thrust::transform(
    thrust::device,
    points,
    points + tree_point_count,
    points,
    [] COMMON (const float3& point)
    {
      return cush::to_cartesian_coords(point);
    });
  cudaDeviceSynchronize();
  
  std::cout << "Normalizing the points." << std::endl;
  for (auto i = 0; i < tree_voxel_count; i++)
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
    points + tree_point_count,
    colors,
    [] COMMON (const float3& point)
    {
      return make_float4(abs(point.x), abs(point.y), abs(point.z), 1.0);
    });
  cudaDeviceSynchronize();

  std::cout << "Translating and scaling the points." << std::endl;
  // TODO: FIX FOR TREE.
  float3 offset       = {
    spacing.x * (block_size.x - 1) * 0.5,
    spacing.y * (block_size.y - 1) * 0.5,
    spacing.z * (block_size.z - 1) * 0.5};
  float3 real_spacing = {
    spacing.x * block_size.x,
    spacing.y * block_size.y,
    spacing.z * block_size.z};
  auto   real_scale   = scale * real_spacing.x * 0.5;
  thrust::transform(
    thrust::device,
    points,
    points + point_count,
    points,
    [=] COMMON (const float3& point)
    {
      auto output = real_scale * point;
      auto index  = int((&point - points) / tessellation_count);
      output.x += offset.x + real_spacing.x * (index / (dimensions.z * dimensions.y));
      output.y += offset.y + real_spacing.y * (index /  dimensions.z % dimensions.y);
      output.z += offset.z + real_spacing.z * (index % dimensions.z);
      return output;
    });
  cudaDeviceSynchronize();

  auto total_end = std::chrono::system_clock::now();
  std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
  std::cout << "Total elapsed time: " << total_elapsed_seconds.count() << "s." << std::endl;
}
}
