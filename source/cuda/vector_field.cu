#include <pli_vis/cuda/vector_field.h>

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

#include <pli_vis/cuda/sh/launch.h>
#include <pli_vis/cuda/sh/vector_ops.h>

namespace pli
{
// Call on a dimensions.x x dimensions.y x dimensions.z 3D grid.
// Vectors are in Cartesian coordinates.
template<typename scalar_type, typename vector_type, typename color_type>
__global__ void create_vector_field_kernel(
  const uint3        dimensions,
  const vector_type* vectors   ,
  const scalar_type  scale     ,
        vector_type* points    ,
        color_type*  colors    )
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= dimensions.x || y >= dimensions.y || z >= dimensions.z)
    return;
  
  auto  volume_index = z + dimensions.z * (y + dimensions.y * x);
  auto& vector       = vectors[volume_index];

  vector_type position = {x, y, z};
  auto        color    = make_float4(abs(vector.x), abs(vector.z), abs(vector.y), 1.0);
  
  auto point_index = 2 * volume_index;
  points[point_index    ] = position + scale * 0.5F * vector;
  points[point_index + 1] = position - scale * 0.5F * vector;
  colors[point_index    ] = color;
  colors[point_index + 1] = color;
}

void create_vector_field(
  const uint3&  dimensions  ,
  const float3* vectors,
  const float&  scale       ,
        float3* points      ,
        float4* colors      ,
  std::function<void(const std::string&)> status_callback)
{
  auto voxel_count = dimensions.x * dimensions.y * dimensions.z;

  thrust::device_vector<float3> gpu_vectors(voxel_count);
  auto gpu_vectors_ptr = raw_pointer_cast(&gpu_vectors[0]);

  copy_n(vectors, voxel_count, gpu_vectors.begin());
  cudaDeviceSynchronize();
  
  create_vector_field_kernel<<<grid_size_3d(dimensions), block_size_3d()>>>(
    dimensions     , 
    gpu_vectors_ptr,
    scale          , 
    points         ,
    colors         );
  cudaDeviceSynchronize();
}
}
