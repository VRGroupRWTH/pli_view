#include /* implements */ <cuda/sampler.h>

#include <chrono>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <cush.h>

namespace pli
{
void sample(
  const uint3&    dimensions       , 
  const unsigned  maximum_degree   ,
  const uint2&    output_resolution, 
  const float*    coefficients     , 
        float3*   points           , 
        unsigned* indices          )
{
  auto total_start = std::chrono::system_clock::now();

  auto voxel_count  = dimensions.x * dimensions.y * dimensions.z;
  auto sample_count = output_resolution.x * output_resolution.y;

  std::cout << "Allocating and copying spherical harmonics coefficients." << std::endl;
  auto coefficients_size = voxel_count * cush::coefficient_count(maximum_degree);
  thrust::device_vector<float> coefficient_vectors(coefficients_size);
  copy_n(coefficients, coefficients_size, coefficient_vectors.begin());
  auto coefficients_ptr = raw_pointer_cast(&coefficient_vectors[0]);

  std::cout << "Allocating points and indices." << std::endl;
  auto points_size  = voxel_count * output_resolution.x * output_resolution.y;
  auto indices_size = 4 * points_size;
  thrust::device_vector<float3>   point_vectors(points_size );
  thrust::device_vector<unsigned> index_vectors(indices_size);
  auto points_ptr  = raw_pointer_cast(&point_vectors[0]);
  auto indices_ptr = raw_pointer_cast(&index_vectors[0]);

  std::cout << "Sampling sums of spherical harmonics coefficients." << std::endl;
  cush::sample_sums<<<dim3(dimensions), 1>>>(
    dimensions, 
    maximum_degree, 
    output_resolution, 
    coefficients_ptr, 
    points_ptr, 
    indices_ptr);

  std::cout << "Normalizing samples." << std::endl;
  //for (auto i = 0; i < voxel_count; i++)
  //{
  //  float3 max_sample = *thrust::max_element(
  //    point_vectors.begin() +  i      * sample_count,
  //    point_vectors.begin() + (i + 1) * sample_count,
  //    [ ] COMMON (const float3& lhs, const float3& rhs)
  //    {
  //      return sqrt(pow(lhs.x, 2) + pow(lhs.y, 2) + pow(lhs.z, 2)) <
  //             sqrt(pow(rhs.x, 2) + pow(rhs.y, 2) + pow(rhs.z, 2));
  //    });
  //  
  //  thrust::transform(
  //    point_vectors.begin() +  i      * sample_count,
  //    point_vectors.begin() + (i + 1) * sample_count,
  //    point_vectors.begin() +  i      * sample_count,
  //    [max_sample] COMMON (float3 value)
  //    {
  //      auto max_sample_length = sqrt(
  //        pow(max_sample.x, 2) +
  //        pow(max_sample.y, 2) +
  //        pow(max_sample.z, 2));
  //      value.x /= max_sample_length;
  //      value.x /= max_sample_length;
  //      value.x /= max_sample_length;
  //      return value;
  //    });
  //}

  std::cout << "Copying points and indices to CPU." << std::endl;
  cudaMemcpy(points , points_ptr , sizeof(float3  ) * points_size , cudaMemcpyDeviceToHost);
  cudaMemcpy(indices, indices_ptr, sizeof(unsigned) * indices_size, cudaMemcpyDeviceToHost);

  auto total_end = std::chrono::system_clock::now();
  std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
  std::cout << "Total elapsed time: " << total_elapsed_seconds.count() << "s." << std::endl;
}
}
