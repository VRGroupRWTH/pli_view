#include <pli_vis/cuda/polar_plot.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <array>
#include <vector>

#include <device_launch_parameters.h>
#include <host_defines.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>

#include <pli_vis/cuda/sh/launch.h>
#include <pli_vis/cuda/utility/convert.h>
#include <pli_vis/cuda/utility/vector_ops.h>

namespace pli
{
template<typename vector_type = float3, typename polar_plot_type = float>
__global__ void calculate_kernel(
  const vector_type* vectors              ,
  const uint2        vectors_dimensions   ,
  const uint2        superpixel_dimensions,
  const unsigned     superpixel_size      ,
  const unsigned     angular_partitions   ,
  const bool         symmetric            ,
  polar_plot_type*   polar_plots          )
{
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= vectors_dimensions.x || y >= vectors_dimensions.y)
    return;

  const auto superpixel_x      = x / superpixel_size;
  const auto superpixel_y      = y / superpixel_size;
  const auto superpixel_offset = (superpixel_y + superpixel_dimensions.y * superpixel_x) * angular_partitions;

  const auto& vector = vectors[y + vectors_dimensions.y * x];
  const auto  index  = static_cast<unsigned>(round((vector.y + (vector.y <= 0.0 ? 2.0 * M_PI : 0.0)) * angular_partitions / (2.0 * M_PI))) % angular_partitions;

  atomicAdd(&polar_plots[superpixel_offset + index], polar_plot_type(1));
  if(symmetric) atomicAdd(&polar_plots[
    superpixel_offset + index + angular_partitions / 2 < angular_partitions ? 
    superpixel_offset + index + angular_partitions / 2                      :
    superpixel_offset + index - angular_partitions / 2], polar_plot_type(1));
}

template<typename polar_plot_type = float, typename vertex_type = float3, typename direction_type = float3>
__global__ void generate_kernel(
  const polar_plot_type* polar_plots          ,
  const uint2            superpixel_dimensions,
  const unsigned         superpixel_size      ,
  const unsigned         angular_partitions   ,
  vertex_type*           vertices             ,
  direction_type*        directions           )
{
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= superpixel_dimensions.x || y >= superpixel_dimensions.y)
    return;

  const auto index             = y + superpixel_dimensions.y * x;
  const auto superpixel_offset = index * angular_partitions;
  const auto vertex_offset     = index * angular_partitions * 3;
  for (auto i = 0, j = 0; j < angular_partitions; i+=3, ++j)
  {
    vertex_type spatial_offset {superpixel_size * (0.5 + x) - 0.5, superpixel_size * (0.5 + y) - 0.5, 0.0};
    vertices  [vertex_offset + i    ] = spatial_offset ,
    vertices  [vertex_offset + i + 1] = spatial_offset + to_cartesian_coords(vertex_type {polar_plots[superpixel_offset + (j + 1) % angular_partitions] * superpixel_size / 2.0, 2.0 * M_PI * ((j + 1) % angular_partitions) / angular_partitions, M_PI / 2.0});
    vertices  [vertex_offset + i + 2] = spatial_offset + to_cartesian_coords(vertex_type {polar_plots[superpixel_offset +  j      % angular_partitions] * superpixel_size / 2.0, 2.0 * M_PI *   j                            / angular_partitions, M_PI / 2.0});
    directions[vertex_offset + i    ] = vertex_type {0, 0, 0};
    directions[vertex_offset + i + 1] = to_cartesian_coords(vertex_type {1, 2.0 * M_PI * (j + 1) / angular_partitions, M_PI / 2.0});
    directions[vertex_offset + i + 2] = to_cartesian_coords(vertex_type {1, 2.0 * M_PI *  j      / angular_partitions, M_PI / 2.0});
  }
}

std::array<std::vector<float3>, 2> calculate(
  const std::vector<float3>& vectors           ,
  const uint2&               vectors_dimensions,
  const unsigned             superpixel_size   ,
  const unsigned             angular_partitions,
  const bool                 symmetric         )
{
  const uint2 superpixel_dimensions {vectors_dimensions.x / superpixel_size, vectors_dimensions.y / superpixel_size};
  const auto  superpixel_count      = superpixel_dimensions.x * superpixel_dimensions.y;

  thrust::device_vector<float3> vectors_gpu(vectors.size());
  thrust::copy(vectors.begin(), vectors.end(), vectors_gpu.begin());
  cudaDeviceSynchronize();

  thrust::device_vector<float> polar_plots_gpu(superpixel_count * angular_partitions);
  calculate_kernel
    <float3, float>
    <<<grid_size_2d(dim3(vectors_dimensions.x, vectors_dimensions.y)), block_size_2d()>>>(
    vectors_gpu    .data().get(),
    vectors_dimensions          ,
    superpixel_dimensions       ,
    superpixel_size             ,
    angular_partitions          ,
    symmetric                   ,
    polar_plots_gpu.data().get());
  cudaDeviceSynchronize();
  
  for (auto i = 0; i < superpixel_count; i++)
  {
    const auto start_iterator = polar_plots_gpu.begin() +  i      * angular_partitions;
    const auto end_iterator   = polar_plots_gpu.begin() + (i + 1) * angular_partitions;
    const auto max_weight     = *max_element(start_iterator, end_iterator);
    thrust::transform(start_iterator, end_iterator, start_iterator, 
    [=] __host__ __device__ (const float& angular_weight) { return angular_weight / max_weight; });
  }
  cudaDeviceSynchronize();
  
  thrust::device_vector<float3> vertices_gpu  (3 * superpixel_count * angular_partitions);
  thrust::device_vector<float3> directions_gpu(3 * superpixel_count * angular_partitions);
  generate_kernel
    <float, float3, float3>
    <<<grid_size_2d(dim3(superpixel_dimensions.x, superpixel_dimensions.y)), block_size_2d()>>>(
    polar_plots_gpu.data().get(),
    superpixel_dimensions       ,
    superpixel_size             ,
    angular_partitions          ,
    vertices_gpu   .data().get(),
    directions_gpu .data().get());
  cudaDeviceSynchronize();
  
  std::array<std::vector<float3>, 2> render_data;
  render_data[0].resize(vertices_gpu  .size());
  render_data[1].resize(directions_gpu.size());
  thrust::copy(vertices_gpu  .begin(), vertices_gpu  .end(), render_data[0].begin());
  thrust::copy(directions_gpu.begin(), directions_gpu.end(), render_data[1].begin());
  cudaDeviceSynchronize();
  return render_data;
}
}
