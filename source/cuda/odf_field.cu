#include /* implements */ <cuda/odf_field.h>

#include <chrono>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <cush.h>

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

  auto voxel_count        = dimensions.x * dimensions.y * dimensions.z;
  auto tessellation_count = tessellations.x * tessellations.y;
  auto coefficients_size  = voxel_count * coefficient_count ;
  auto point_count        = voxel_count * tessellation_count;

  std::cout << "Allocating and copying spherical harmonics coefficients." << std::endl;
  thrust::device_vector<float> coefficient_vectors(coefficients_size);
  copy_n(coefficients, coefficients_size, coefficient_vectors.begin());
  auto coefficients_ptr = raw_pointer_cast(&coefficient_vectors[0]);

  std::cout << "Sampling sums of spherical harmonics coefficients." << std::endl;
  cush::sample_sums<<<dim3(dimensions), 1>>>(
    dimensions       ,
    coefficient_count,
    tessellations    ,
    coefficients_ptr , 
    points           , 
    indices          );
  cudaDeviceSynchronize();

  std::cout << "Converting samples to Cartesian coordinates." << std::endl;
  thrust::transform(
    thrust::device,
    points,
    points + point_count,
    points,
    [] COMMON (const float3& point)
    {
      return cush::to_cartesian_coords<float3>(point);
    });
  cudaDeviceSynchronize();
  
  //std::cout << "Normalizing samples." << std::endl;
  //for (auto i = 0; i < voxel_count; i++)
  //{
  //  float3* max_sample = thrust::max_element(
  //    thrust::device,
  //    points +  i      * tessellation_count,
  //    points + (i + 1) * tessellation_count,
  //    [] COMMON (const float3& lhs, const float3& rhs)
  //    {
  //      return sqrt(pow(lhs.x, 2) + pow(lhs.y, 2) + pow(lhs.z, 2)) <
  //             sqrt(pow(rhs.x, 2) + pow(rhs.y, 2) + pow(rhs.z, 2));
  //    });
  //
  //  thrust::transform(
  //    thrust::device,
  //    points +  i      * tessellation_count,
  //    points + (i + 1) * tessellation_count,
  //    points +  i      * tessellation_count,
  //    [max_sample] COMMON(float3 value)
  //    {
  //      auto max_sample_length = sqrt(
  //        pow(max_sample->x, 2) +
  //        pow(max_sample->y, 2) +
  //        pow(max_sample->z, 2));
  //
  //      value.x /= max_sample_length;
  //      value.y /= max_sample_length;
  //      value.z /= max_sample_length;
  //      return value;
  //    });
  //}
  //cudaDeviceSynchronize();

  auto total_end = std::chrono::system_clock::now();
  std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
  std::cout << "Total elapsed time: " << total_elapsed_seconds.count() << "s." << std::endl;
}
}
