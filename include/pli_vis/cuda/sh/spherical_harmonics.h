#ifndef PLI_VIS_SPHERICAL_HARMONICS_H_
#define PLI_VIS_SPHERICAL_HARMONICS_H_

#define _USE_MATH_DEFINES

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <vector_types.h>
#include <thrust/sort.h>

#include <pli_vis/cuda/sh/clebsch_gordan.h>
#include <pli_vis/cuda/sh/factorial.h>
#include <pli_vis/cuda/sh/launch.h>
#include <pli_vis/cuda/sh/legendre.h>

// Based on "Spherical Harmonic Lighting: The Gritty Details" by Robin Green.
namespace pli
{
__forceinline__ __host__ __device__ unsigned int maximum_degree   (const unsigned int coefficient_count)
{
  return unsigned(sqrtf(float(coefficient_count)) - 1);
}
__forceinline__ __host__ __device__ unsigned int coefficient_count(const unsigned int max_l)
{
  return (max_l + 1) * (max_l + 1);
}
__forceinline__ __host__ __device__ unsigned int coefficient_index(const unsigned int l, const int m)
{
  return l * (l + 1) + m;
}
__forceinline__ __host__ __device__ int2         coefficient_lm   (const unsigned int index)
{
  int2 lm;
  lm.x = int(floor(sqrtf(float(index))));
  lm.y = int(index - powf(lm.x, 2) - lm.x);
  return lm;
}

template<typename precision>
__host__ __device__ precision evaluate(
  const unsigned int l    ,
  const          int m    ,
  const precision&   theta,
  const precision&   phi  )
{
  precision kml = sqrt((2.0 * l + 1) * factorial<precision>(l - abs(m)) / 
                       (4.0 * M_PI   * factorial<precision>(l + abs(m))));
  if (m > 0)
    return sqrt(2.0) * kml * cos( m * theta) * associated_legendre(l,  m, cos(phi));
  if (m < 0)
    return sqrt(2.0) * kml * sin(-m * theta) * associated_legendre(l, -m, cos(phi));
  return kml * associated_legendre(l, 0, cos(phi));
}
template<typename precision>
__host__ __device__ precision evaluate(
  const unsigned int index,
  const precision&   theta,
  const precision&   phi  )
{
  auto lm = coefficient_lm(index);
  return evaluate(lm.x, lm.y, theta, phi);
}

// Not used internally as the two for loops can also be further parallelized.
template<typename precision>
__host__ __device__ precision evaluate_sum(
  const unsigned int max_l       ,
  const precision&   theta       ,
  const precision&   phi         ,
  const precision*   coefficients)
{
  precision sum = 0.0;
  for (int l = 0; l <= max_l; l++)
    for (int m = -l; m <= l; m++)
      sum += evaluate(l, m, theta, phi) * coefficients[coefficient_index(l, m)];
  return sum;
}

template<typename precision>
__host__ __device__ precision is_zero(
  const unsigned int coefficient_count,
  const precision*   coefficients )
{
  for (auto index = 0; index < coefficient_count; index++)
    if (coefficients[index] != precision(0))
      return false;
  return true;
}

template<typename precision>
__host__ __device__ precision l1_distance(
  const unsigned int coefficient_count,
  const precision*   lhs_coefficients ,
  const precision*   rhs_coefficients )
{
  precision value = 0;
  for (auto index = 0; index < coefficient_count; index++)
    value += abs(lhs_coefficients[index] - rhs_coefficients[index]);
  return value;
}

// Based on "Rotation Invariant Spherical Harmonic Representation of 3D Shape Descriptors" by Kazhdan et al.
template<typename precision>
__host__ __device__ precision l2_distance(
  const unsigned int coefficient_count,
  const precision*   lhs_coefficients ,
  const precision*   rhs_coefficients )
{
  precision value = 0;
  for (auto index = 0; index < coefficient_count; index++)
    value += pow(lhs_coefficients[index] - rhs_coefficients[index], 2);
  return sqrt(value);
}

// Call on a vector_count x coefficient_count(max_l) 2D grid.
template<typename vector_type, typename precision>
__global__ void calculate_matrix(
  const unsigned int vector_count     ,
  const unsigned int coefficient_count,
  const vector_type* vectors          , 
  precision*         output_matrix    ,
  bool               even_only        = true)
{
  auto vector_index      = blockIdx.x * blockDim.x + threadIdx.x;
  auto coefficient_index = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (vector_index      >= vector_count     || 
      coefficient_index >= coefficient_count)
    return;

  if(!even_only || coefficient_index % 2 == 0)
    atomicAdd(
      &output_matrix[vector_index + vector_count * coefficient_index], 
      evaluate(coefficient_index, vectors[vector_index].y, vectors[vector_index].z));
}
// Call on a dimensions.x x dimensions.y x dimensions.z 3D grid.
template<typename vector_type, typename precision>
__global__ void calculate_matrices(
  const uint3        dimensions       ,
  const unsigned int vector_count     , 
  const unsigned int coefficient_count,
  const vector_type* vectors          ,
  precision*         output_matrices  )
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x >= dimensions.x || y >= dimensions.y || z >= dimensions.z)
    return;
  
  auto vectors_offset = vector_count  * (z + dimensions.z * (y + dimensions.y * x));
  auto matrix_offset  = vectors_offset * coefficient_count;
  
  calculate_matrix<<<grid_size_2d(dim3(vector_count, coefficient_count)), block_size_2d()>>>(
    vector_count     , 
    coefficient_count, 
    vectors         + vectors_offset, 
    output_matrices + matrix_offset );
}

// Call on a tessellations.x x tessellations.y 2D grid.
template<typename point_type>
__global__ void sample(
  const unsigned int l             ,
  const int          m             ,
  const uint2        tessellations ,
  point_type*        output_points ,
  unsigned int*      output_indices)
{
  auto longitude = blockIdx.x * blockDim.x + threadIdx.x;
  auto latitude  = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (longitude >= tessellations.x ||
      latitude  >= tessellations.y )
    return;
  
  auto point_offset = latitude + longitude * tessellations.y;
  auto index_offset = 6 * point_offset;

  auto& point = output_points[point_offset];
  point.y = 2 * M_PI * longitude /  tessellations.x;
  point.z =     M_PI * latitude  / (tessellations.y - 1);
  point.x = evaluate(l, m, point.y, point.z);

  output_indices[index_offset    ] =  longitude                        * tessellations.y +  latitude,
  output_indices[index_offset + 1] =  longitude                        * tessellations.y + (latitude + 1) % tessellations.y,
  output_indices[index_offset + 2] = (longitude + 1) % tessellations.x * tessellations.y + (latitude + 1) % tessellations.y,
  output_indices[index_offset + 3] =  longitude                        * tessellations.y +  latitude,
  output_indices[index_offset + 4] = (longitude + 1) % tessellations.x * tessellations.y + (latitude + 1) % tessellations.y,
  output_indices[index_offset + 5] = (longitude + 1) % tessellations.x * tessellations.y +  latitude;
}
// Call on a tessellations.x x tessellations.y x coefficient_count(max_l) 3D grid.
template<typename precision, typename point_type>
__global__ void sample_sum(
  const unsigned int coefficient_count   ,
  const uint2        tessellations       ,
  const precision*   coefficients        ,
  point_type*        output_points       ,
  unsigned int*      output_indices      = nullptr,
  const unsigned int base_index          = 0      )
{
  auto longitude         = blockIdx.x * blockDim.x + threadIdx.x;
  auto latitude          = blockIdx.y * blockDim.y + threadIdx.y;
  auto coefficient_index = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (longitude         >= tessellations.x  ||
      latitude          >= tessellations.y  ||
      coefficient_index >= coefficient_count)
    return;

  auto  point_offset = latitude + longitude * tessellations.y;
  auto& point        = output_points[point_offset];

  if (coefficient_index == 0)
    point.x = 0;
  
  point.y = 2 * M_PI * longitude /  tessellations.x;
  point.z =     M_PI * latitude  / (tessellations.y - 1);
  atomicAdd(&point.x, evaluate(coefficient_index, point.y, point.z) * coefficients[coefficient_index]);

  if (output_indices != nullptr && coefficient_index == 0)
  {
    auto index_offset = 6 * point_offset;
    output_indices[index_offset    ] = base_index +  longitude                        * tessellations.y +  latitude,
    output_indices[index_offset + 1] = base_index +  longitude                        * tessellations.y + (latitude + 1) % tessellations.y,
    output_indices[index_offset + 2] = base_index + (longitude + 1) % tessellations.x * tessellations.y + (latitude + 1) % tessellations.y,
    output_indices[index_offset + 3] = base_index +  longitude                        * tessellations.y +  latitude,
    output_indices[index_offset + 4] = base_index + (longitude + 1) % tessellations.x * tessellations.y + (latitude + 1) % tessellations.y,
    output_indices[index_offset + 5] = base_index + (longitude + 1) % tessellations.x * tessellations.y +  latitude;
  }
}
// Call on a dimensions.x x dimensions.y x dimensions.z 3D grid.
template<typename precision, typename point_type>
__global__ void sample_sums(
  const uint3        dimensions         ,
  const unsigned int coefficient_count  ,
  const uint2        tessellations      ,
  const precision*   coefficients       ,
  point_type*        output_points      ,
  unsigned int*      output_indices     ,
  const unsigned int base_index         = 0   ,
  const bool         normalize          = true)
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x >= dimensions.x || 
      y >= dimensions.y || 
      z >= dimensions.z )
    return;
  
  auto volume_index        = z + dimensions.z * (y + dimensions.y * x);
  auto coefficients_offset = volume_index * coefficient_count;
  auto points_size         = tessellations.x * tessellations.y;
  auto points_offset       = volume_index * points_size;
  auto indices_offset      = 6 * points_offset;

  sample_sum<<<grid_size_3d(dim3(tessellations.x, tessellations.y, coefficient_count)), block_size_3d()>>>(
    coefficient_count,
    tessellations    ,
    coefficients   + coefficients_offset, 
    output_points  + points_offset      ,
    output_indices + indices_offset     ,
    base_index     + points_offset      );
  
  cudaDeviceSynchronize();

  if (normalize)
  {
    auto maxima = 0.0;
    for (auto i = 0; i < points_size; i++)
      if (maxima < output_points[points_offset + i].x)
        maxima = output_points[points_offset + i].x;
    for (auto i = 0; i < points_size; i++)
      output_points[points_offset + i].x = output_points[points_offset + i].x / maxima;
  }
}
// Call on a dimensions.x x dimensions.y x dimensions.z 3D grid.
template<typename precision, typename vector_type>
__global__ void extract_maxima(
  // Input data parameters.
  const uint3        dimensions       ,
  const unsigned int coefficient_count,
  const precision*   coefficients     ,
  // Peak extraction parameters.
  const uint2        tessellations    ,
  const unsigned int maxima_count     ,
  // Output data parameters.
  vector_type*       maxima           )
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x >= dimensions.x || 
      y >= dimensions.y || 
      z >= dimensions.z )
    return;
  
  auto volume_index        = z + dimensions.z * (y + dimensions.y * x);
  auto coefficients_offset = volume_index * coefficient_count;
  auto points_size         = tessellations.x * tessellations.y;

  float3* points;
  cudaMalloc(reinterpret_cast<void**>(&points), points_size * sizeof(float3));

  sample_sum<<<grid_size_3d(dim3(tessellations.x, tessellations.y, coefficient_count)), block_size_3d()>>>(
    coefficient_count                 ,
    tessellations                     ,
    coefficients + coefficients_offset,
    points                            );
  cudaDeviceSynchronize();

  thrust::sort(thrust::seq, points, points + points_size, 
  [&] (const float3& lhs, const float3& rhs)
  {
    return lhs.x > rhs.x;
  });

  for(auto i = 0; i < maxima_count; i++)
    maxima[volume_index * maxima_count + i] = points[i];

  cudaFree(points);
}

// Call on a coefficient_count x coefficient_count x coefficient_count 3D grid.
// Based on Modern Quantum Mechanics 2nd Edition page 216 by Jun John Sakurai.
template<typename precision, typename atomics_precision = float>
__global__ void product(
  const unsigned int coefficient_count,
  const precision*   lhs_coefficients ,
  const precision*   rhs_coefficients ,
  atomics_precision* out_coefficients )
{
  auto lhs_index = blockIdx.x * blockDim.x + threadIdx.x;
  auto rhs_index = blockIdx.y * blockDim.y + threadIdx.y;
  auto out_index = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (lhs_index >= coefficient_count ||
      rhs_index >= coefficient_count ||
      out_index >= coefficient_count)
    return;

  auto lhs_lm   = coefficient_lm(lhs_index);
  auto rhs_lm   = coefficient_lm(rhs_index);
  auto out_lm   = coefficient_lm(out_index);
  auto cg1      = clebsch_gordan<atomics_precision>(lhs_lm.x, rhs_lm.x, out_lm.x, 0, 0, 0);
  auto cg2      = clebsch_gordan<atomics_precision>(lhs_lm.x, rhs_lm.x, out_lm.x, lhs_lm.y, rhs_lm.y, out_lm.y);
  auto coupling = sqrt((2 * lhs_lm.x + 1) * (2 * rhs_lm.x + 1) / (4 * M_PI * (2 * out_lm.x + 1))) * cg1 * cg2;

  atomicAdd(&out_coefficients[out_index], atomics_precision(coupling * lhs_coefficients[lhs_index] * rhs_coefficients[rhs_index]));
}
// Call on a dimensions.x x dimensions.y x dimensions.z 3D grid.
template<typename precision, typename atomics_precision = float>
__global__ void product(
  const uint3        dimensions       ,
  const unsigned int coefficient_count,
  const precision*   lhs_coefficients ,
  const precision*   rhs_coefficients ,
  atomics_precision* out_coefficients )
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= dimensions.x || y >= dimensions.y || z >= dimensions.z)
    return;

  auto coefficients_offset = coefficient_count * (z + dimensions.z * (y + dimensions.y * x));
  
  product<<<grid_size_3d(dim3(coefficient_count, coefficient_count, coefficient_count)), block_size_3d()>>>(
    coefficient_count,
    lhs_coefficients + coefficients_offset,
    rhs_coefficients + coefficients_offset,
    out_coefficients + coefficients_offset);
}
}

#endif
