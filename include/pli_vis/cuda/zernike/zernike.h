#ifndef ZERNIKE_H_
#define ZERNIKE_H_

#include <math.h>

#include <device_launch_parameters.h>
#include <host_defines.h>
#include <vector_types.h>

// References:
// - Lakshminarayanan & Fleck, Zernike Polynomials: A Guide, Journal of Modern Optics 2011.7.
// Notes:
// - Rho   is restricted to the unit circle.
// - Theta is measured clockwise from the vertical axis and is in radians.
namespace zer
{
// Launch utility.
__forceinline__ __host__ __device__ unsigned block_size_1d()
{
  return 64;
}
__forceinline__ __host__ __device__ dim3     block_size_2d()
{
  return {32, 32, 1};
}
__forceinline__ __host__ __device__ dim3     block_size_3d()
{
  return {16, 16, 4};
}
__forceinline__ __host__ __device__ unsigned grid_size_1d (const unsigned& target_dimension )
{
  const auto block_size = block_size_1d();
  return (target_dimension + block_size - 1u) / block_size;
}
__forceinline__ __host__ __device__ dim3     grid_size_2d (const dim3&     target_dimensions)
{
  const auto block_size = block_size_2d();
  return 
  {
    (target_dimensions.x + block_size.x - 1u) / block_size.x,
    (target_dimensions.y + block_size.y - 1u) / block_size.y,
    1u
  };
}
__forceinline__ __host__ __device__ dim3     grid_size_3d (const dim3&     target_dimensions)
{
  const auto block_size = block_size_3d();
  return 
  {
    (target_dimensions.x + block_size.x - 1) / block_size.x,
    (target_dimensions.y + block_size.y - 1) / block_size.y,
    (target_dimensions.z + block_size.z - 1) / block_size.z
  };
}

// Factorial utility.
template<typename precision>
__host__ __device__ precision factorial(unsigned n)
{
  precision out(1);
  for (auto i = 2; i <= n; i++)
    out *= i;
  return out;
}

// Zernike moments implementation.
__forceinline__ __host__ __device__ unsigned maximum_degree(const unsigned& count)
{
  return round(sqrt(2 * count) - 1);
}
__forceinline__ __host__ __device__ unsigned expansion_size(const unsigned& max_n)
{
  return (max_n + 1) * (max_n + 2) / 2;
}
__forceinline__ __host__ __device__ unsigned linear_index  (const int2&     nm   )
{
  return (nm.x * (nm.x + 2) + nm.y) / 2;
}
__forceinline__ __host__ __device__ int2     quantum_index (const unsigned& i    )
{
  int2 nm;
  nm.x = ceil((-3 + sqrt(9 + 8 * i)) / 2);
  nm.y = 2 * i - nm.x * (nm.x + 2);
  return nm;
}

template<typename precision>
__host__ __device__ precision mode    (const int2& nm, const precision& rho)
{
  precision out(0);
  for(unsigned i = 0; i < (nm.x - nm.y) / 2; i++)
    out += pow(rho, nm.x - 2 * i) * 
     (pow(-1, i) * factorial<precision>(nm.x - i)) / 
     (factorial<precision>(i) * 
      factorial<precision>(0.5 * (nm.x + nm.y) - i) * 
      factorial<precision>(0.5 * (nm.x - nm.y) - i));
  return out;
}
template<typename precision>
__host__ __device__ precision evaluate(const int2& nm, const precision& rt )
{
  return mode(nm, rt.x) * (nm.y >= 0 ? cos(nm.y * rt.y) : sin(nm.y * rt.y));
}

// This kernel requires a samples_per_voxel x expansion_size 2D grid.
template<typename precision, typename sample_type>
__global__ void compute_basis(
  const unsigned     sample_count     , 
  const sample_type* samples          ,
  const unsigned     expansion_size   ,
  precision*         basis            )
{
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= sample_count || y >= expansion_size)
    return;

  atomicAdd(&basis[x + sample_count * y], evaluate(quantum_index(y), samples[x]));
}
// This kernel requires a dimensions.x x dimensions.y x dimensions.z 3D grid.
template<typename precision, typename sample_type>
__global__ void compute_bases(
  const uint3        dimensions       ,
  const unsigned     samples_per_voxel, 
  const sample_type* samples          ,
  const unsigned     expansion_size   ,
  precision*         bases            )
{
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  const auto z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= dimensions.x || y >= dimensions.y || z >= dimensions.z)
    return;

  const auto samples_offset = samples_per_voxel * (z + dimensions.z * (y + dimensions.y * x));
  const auto bases_offset   = samples_offset    * expansion_size;
  compute_basis<<<grid_size_2d(dim3(samples_per_voxel, expansion_size)), block_size_2d()>>>(
    samples_per_voxel       ,
    samples + samples_offset,
    expansion_size          , 
    bases   + bases_offset  );
}

// This kernel requires a expansion_size x samples_per_voxel 2D grid.
template<typename precision, typename coefficient_type>
__global__ void reconstruct(
  const unsigned          expansion_size   ,
  const coefficient_type* coefficients     ,
  const unsigned          samples_per_voxel,
  const precision*        samples          ,
  precision*              outputs          )
{
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= expansion_size || y >= samples_per_voxel)
    return;

  atomicAdd(&outputs[y], coefficients[x] * evaluate(x, samples[y]));
}
// This kernel requires a dimensions.x x dimensions.y x dimensions.z 3D grid.
template<typename precision, typename coefficient_type>
__global__ void reconstruct(
  const uint3             dimensions       ,
  const unsigned          expansion_size   ,
  const coefficient_type* coefficients     ,
  const unsigned          samples_per_voxel,
  const precision*        samples          ,
  precision*              outputs          )
{
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  const auto z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= dimensions.x || y >= dimensions.y || z >= dimensions.z)
    return;
  
  const auto volume_index        = z + dimensions.z * (y + dimensions.y * x);
  const auto coefficients_offset = volume_index * expansion_size   ;
  const auto samples_offset      = volume_index * samples_per_voxel;
  reconstruct<<<grid_size_2d(dim3(expansion_size, samples_per_voxel)), block_size_2d()>>>(
    expansion_size                    ,
    coefficients + coefficients_offset, 
    samples_per_voxel                 ,
    samples      + samples_offset     ,
    outputs      + samples_offset     );
}
}

#endif
