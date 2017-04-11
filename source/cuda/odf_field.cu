#include /* implements */ <cuda/odf_field.h>

#include <chrono>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <cush.h>
#include <vector_ops.h>

#include <cuda/launch.h>

namespace pli
{
void create_odfs(
  const uint3&    dimensions       ,
  const unsigned  coefficient_count,
  const float*    coefficients     ,
  const uint2&    tessellations    ,
  const float3&   vector_spacing   ,
  const uint3&    vector_dimensions,
  const float     scale            ,
        float3*   points           ,
        float4*   colors           ,
        unsigned* indices          ,
        bool      clustering       ,
        float     cluster_threshold,
        std::function<void(const std::string&)> status_callback)
{
  auto total_start = std::chrono::system_clock::now();
  
  auto base_voxel_count = dimensions.x * dimensions.y * dimensions.z;
  auto dimension_count  = dimensions.z > 1 ? 3 : 2;
  auto min_dimension    = min(dimensions.x, dimensions.y);
  if (dimension_count == 3)
    min_dimension = min(min_dimension, dimensions.z);
  auto max_layer        = log(min_dimension) / log(2);
  auto voxel_count      = unsigned(base_voxel_count * 
    ((1.0 - pow(1.0 / pow(2, dimension_count), max_layer + 1)) / 
     (1.0 -     1.0 / pow(2, dimension_count))));

  auto tessellation_count = tessellations.x * tessellations.y;
  auto point_count        = voxel_count * tessellation_count;

  status_callback("Allocating and copying the leaf spherical harmonics coefficients.");
  thrust::device_vector<float> coefficient_vectors(voxel_count * coefficient_count);
  copy_n(coefficients, base_voxel_count * coefficient_count, coefficient_vectors.begin());
  auto coefficients_ptr = raw_pointer_cast(&coefficient_vectors[0]);

  auto layer_offset     = 0;
  auto layer_dimensions = dimensions;
  for (auto layer = max_layer; layer >= 0; layer--)
  {
    if (layer != max_layer)
    {
      status_callback("Calculating the layer " + std::to_string(int(layer)) + " coefficients.");
      create_layer<<<grid_size_3d(layer_dimensions), block_size_3d()>>>(
        layer_dimensions    ,
        layer_offset        ,
        coefficient_count   ,
        coefficients_ptr    ,
        dimension_count == 2,
        clustering          ,
        cluster_threshold   );
      cudaDeviceSynchronize();
    }
    
    layer_offset += layer_dimensions.x * layer_dimensions.y * layer_dimensions.z;
    layer_dimensions = {
      layer_dimensions.x / 2,
      layer_dimensions.y / 2,
      dimension_count == 3 ? layer_dimensions.z / 2 : 1
    };
  }
  
  layer_offset     = 0;
  layer_dimensions = dimensions;
  for (auto layer = max_layer; layer >= 0; layer--)
  {
    status_callback("Sampling sums of the layer " + std::to_string(int(layer)) + " coefficients.");
    cush::sample_sums<<<grid_size_3d(layer_dimensions), block_size_3d()>>>(
      layer_dimensions ,
      coefficient_count,
      tessellations    ,
      coefficients_ptr + layer_offset * coefficient_count , 
      points           + layer_offset * tessellation_count, 
      indices          + layer_offset * tessellation_count * 6,
      layer_offset     * tessellation_count);
    cudaDeviceSynchronize();
    
    layer_offset += layer_dimensions.x * layer_dimensions.y * layer_dimensions.z;
    layer_dimensions = {
      layer_dimensions.x / 2,
      layer_dimensions.y / 2,
      dimension_count == 3 ? layer_dimensions.z / 2 : 1
    };
  }

  status_callback("Converting the points to Cartesian coordinates.");
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

  status_callback("Assigning colors.");
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

  status_callback("Translating and scaling the points.");
  layer_offset     = 0;
  layer_dimensions = dimensions;
  for (auto layer = max_layer; layer >= 0; layer--)
  {
    auto layer_point_offset = 
      layer_offset * 
      tessellation_count;
    auto layer_point_count  =  
      layer_dimensions.x * 
      layer_dimensions.y * 
      layer_dimensions.z * 
      tessellation_count;

    uint3 layer_vectors_size {
      vector_dimensions.x * pow(2, max_layer - layer),
      vector_dimensions.y * pow(2, max_layer - layer),
      vector_dimensions.z * pow(2, max_layer - layer) };
    float3 layer_position {
      vector_spacing.x * (layer_vectors_size.x - 1) * 0.5,
      vector_spacing.y * (layer_vectors_size.y - 1) * 0.5,
      dimension_count == 3 ? vector_spacing.z * (layer_vectors_size.z - 1) * 0.5 : 0.0 };
    float3 layer_spacing {
      vector_spacing.x * layer_vectors_size.x,
      vector_spacing.y * layer_vectors_size.y,
      dimension_count == 3 ? vector_spacing.z * layer_vectors_size.z : 1.0 };
    auto layer_scale = scale * min(min(layer_spacing.x, layer_spacing.y), layer_spacing.z) * 0.5;

    thrust::transform(
      thrust::device,
      points + layer_point_offset,
      points + layer_point_offset + layer_point_count,
      points + layer_point_offset,
      [=] COMMON (const float3& point)
      {
        auto output = layer_scale * point;
        auto index  = int((&point - (points + layer_point_offset)) / tessellation_count);
        output.x += layer_position.x + layer_spacing.x * (index / (layer_dimensions.z * layer_dimensions.y));
        output.y += layer_position.y + layer_spacing.y * (index /  layer_dimensions.z % layer_dimensions.y);
        output.z += layer_position.z + layer_spacing.z * (index % layer_dimensions.z);
        return output;
      });
    cudaDeviceSynchronize();

    layer_offset += layer_dimensions.x * layer_dimensions.y * layer_dimensions.z;
    layer_dimensions = {
      layer_dimensions.x / 2,
      layer_dimensions.y / 2,
      dimension_count == 3 ? layer_dimensions.z / 2 : 1
    };
  }

  status_callback("Inverting Y coordinates.");
  thrust::transform(
    thrust::device,
    points,
    points + point_count,
    points,
    [] COMMON(float3 point)
  {
    point.y = -point.y;
    return point;
  });
  cudaDeviceSynchronize();

  auto total_end = std::chrono::system_clock::now();
  std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
  status_callback("Cuda operations took " + std::to_string(total_elapsed_seconds.count()) + " seconds.");
}
}
