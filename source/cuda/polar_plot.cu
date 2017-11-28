#include <pli_vis/cuda/polar_plot.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <device_launch_parameters.h>
#include <host_defines.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>

#include <pli_vis/cuda/sh/launch.h>

namespace pli
{
template<typename vector_type = float3, typename polar_plot_type = float>
__global__ void calculate_kernel(
  const vector_type* vectors           ,
  polar_plot_type*   polar_plots       ,
  const uint2&       vectors_dimensions,
  const unsigned     superpixel_size   ,
  const unsigned     angular_partitions,
  const bool         symmetric         )
{
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= vectors_dimensions.x || y >= vectors_dimensions.y)
    return;

  const auto superpixel_x      = x / superpixel_size;
  const auto superpixel_y      = y / superpixel_size;
  const auto superpixel_offset = (superpixel_y + superpixel_size * superpixel_x) * angular_partitions;

  const auto& vector = vectors[y + superpixel_size * x];
  const auto  index  = static_cast<unsigned>(round(vector.y / (2.0 * M_PI / angular_partitions)));

                atomicAdd(&polar_plots[superpixel_offset + index]                         , polar_plot_type(1));
  if(symmetric) atomicAdd(&polar_plots[superpixel_offset + index + angular_partitions / 2], polar_plot_type(1));
}

template<typename polar_plot_type = float, typename vertex_type = float3>
__global__ void generate_kernel()
{
  
}

std::vector<float> calculate(
  const std::vector<float3>& vectors           ,
  const uint2&               vectors_dimensions,
  const unsigned             superpixel_size   ,
  const unsigned             angular_partitions,
  const bool                 symmetric         )
{
  const uint2 superpixel_dimensions {vectors_dimensions.x / superpixel_size, vectors_dimensions.y / superpixel_size};
  const auto  superpixel_count      = superpixel_dimensions.x * superpixel_dimensions.y;

  thrust::device_vector<float3> vectors_gpu    (vectors.size());
  thrust::device_vector<float>  polar_plots_gpu(superpixel_count * angular_partitions);
  calculate_kernel
    <float3, float>
    <<<grid_size_2d(dim3(vectors_dimensions.x, vectors_dimensions.y)), block_size_2d()>>>(
    vectors_gpu    .data().get(),
    polar_plots_gpu.data().get(),
    vectors_dimensions          ,
    superpixel_size             ,
    angular_partitions          ,
    symmetric                   );
  cudaDeviceSynchronize();

  for (auto i = 0; i < superpixel_count; i++)
  {
    const auto start_iterator  = polar_plots_gpu.begin() +  i      * angular_partitions;
    const auto end_iterator    = polar_plots_gpu.begin() + (i + 1) * angular_partitions;
    const auto max_weight      = *max_element(start_iterator, end_iterator);
    thrust::transform(start_iterator, end_iterator, start_iterator, 
    [=] __host__ __device__ (const float& angular_weight) { return angular_weight / max_weight; });
  }
  cudaDeviceSynchronize();

  thrust::device_vector<float3>   vertices_gpu  (superpixel_count * angular_partitions);
  thrust::device_vector<float3>   directions_gpu(superpixel_count * angular_partitions);
  thrust::device_vector<unsigned> indices_gpu   (superpixel_count * (angular_partitions + 1));
  generate_kernel
    <float, float3>
    <<<grid_size_2d(dim3(vectors_dimensions.x / superpixel_size, vectors_dimensions.y / superpixel_size)), block_size_2d()>>>(
    superpixel_size   ,
    angular_partitions);

  std::vector<float> polar_plots(polar_plots_gpu.size());
  thrust::copy(polar_plots_gpu.begin(), polar_plots_gpu.end(), polar_plots.begin());
  return polar_plots;
}
}
