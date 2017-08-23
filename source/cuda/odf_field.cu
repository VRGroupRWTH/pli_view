#include <pli_vis/cuda/odf_field.h>

#include <chrono>
#include <math.h>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <pli_vis/cuda/sh/launch.h>
#include <pli_vis/cuda/sh/spherical_harmonics.h>
#include <pli_vis/cuda/utility/convert.h>
#include <pli_vis/cuda/utility/vector_ops.h>
#include <pli_vis/cuda/spherical_histogram.h>

namespace pli
{
// Call on a dimensions.x x dimensions.y x dimensions.z 3D grid.
// Vectors are in spherical coordinates.
template<typename vector_type, typename scalar_type>
__global__ void quantify_and_project_kernel(
  // Input related parameters.
        uint3          dimensions         ,
        uint3          vectors_dimensions ,
  // Quantification related parameters. 
  const vector_type*   vectors            , 
        unsigned       histogram_bin_count,
        vector_type*   histogram_vectors  ,
  // Projection related parameters.     
        unsigned       maximum_degree     ,
  const scalar_type*   inverse_transform  ,
        scalar_type*   coefficient_vectors)
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= dimensions.x || y >= dimensions.y || z >= dimensions.z)
    return;
    
  auto volume_index              = z + dimensions.z * (y + dimensions.y * x);
  auto coefficient_count         = pli::coefficient_count(maximum_degree);
  auto coefficient_vector_offset = volume_index * coefficient_count;
  auto alpha                     = 1.0F;
  auto beta                      = 0.0F;
  
  float* histogram_magnitudes;
  cudaMalloc(&histogram_magnitudes, histogram_bin_count * sizeof(float));
  for(auto i = 0; i < histogram_bin_count; ++i)
    histogram_magnitudes[i] = 0.0F;
    
  uint3 offset {
    vectors_dimensions.x * x,
    vectors_dimensions.y * y,
    vectors_dimensions.z * z };
  uint3 size {
    vectors_dimensions.x * dimensions.x,
    vectors_dimensions.y * dimensions.y,
    vectors_dimensions.z * dimensions.z };

  cublasHandle_t cublas;
  cublasCreate(&cublas);

  accumulate_kernel<<<grid_size_3d(vectors_dimensions), block_size_3d()>>>(
    vectors_dimensions  ,
    offset              ,
    size                ,
    vectors             ,
    histogram_bin_count ,
    histogram_vectors   ,
    histogram_magnitudes);
  __syncthreads();
  cudaDeviceSynchronize();
  __syncthreads();

  cublasSgemv(
    cublas                                         ,
    CUBLAS_OP_N                                    ,
    coefficient_count                              ,
    histogram_bin_count                            ,
    &alpha                                         ,
    inverse_transform                              ,
    coefficient_count                              ,
    histogram_magnitudes                           ,
    1                                              ,
    &beta                                          ,
    coefficient_vectors + coefficient_vector_offset,
    1                                              );

  cublasDestroy(cublas);

  cudaFree(histogram_magnitudes);
}

// Call on a dimensions.x x dimensions.y x dimensions.z 3D grid.
__global__ void sum_and_cluster_kernel(
  // Sum related parameters. 
  const uint3    dimensions       , 
  const unsigned offset           ,
  unsigned       maximum_degree   ,
  float*         coefficients     , 
  bool           limit_to_2d      ,
  // Clustering related parameters.     
  bool           clustering       , 
  float          cluster_threshold)
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= dimensions.x || y >= dimensions.y || z >= dimensions.z)
    return;

  uint3 lower_layer_dimensions = {dimensions.x * 2, dimensions.y * 2, limit_to_2d ? 1 : dimensions.z * 2};
  auto  lower_layer_offset     = offset - lower_layer_dimensions.x * lower_layer_dimensions.y * lower_layer_dimensions.z;
  auto  coefficient_count      = pli::coefficient_count(maximum_degree);
  auto  coefficient_offset     = coefficient_count * (offset + z + dimensions.z * (y + dimensions.y * x));

  for (auto i = 0; i < 2; i++)
    for (auto j = 0; j < 2; j++)
      for (auto k = 0; k < (limit_to_2d ? 1 : 2); k++)
        for (auto c = 0; c < coefficient_count; c++)
          coefficients[coefficient_offset + c] += coefficients[coefficient_count * (lower_layer_offset + (2 * z + k) + lower_layer_dimensions.z * (2 * y + j + lower_layer_dimensions.y * (2 * x + i))) + c] / powf(2, limit_to_2d ? 2 : 3);

  if (clustering)
  {
    // Compare this voxel to each associated voxel. 
    auto is_similar = true;
    for (auto i = 0; i < 2; i++)
      for (auto j = 0; j < 2; j++)
        for (auto k = 0; k < (limit_to_2d ? 1 : 2); k++)
        {
          auto other_offset = coefficient_count * (lower_layer_offset + (2 * z + k) + lower_layer_dimensions.z * (2 * y + j + lower_layer_dimensions.y * (2 * x + i)));
          if (is_zero(coefficient_count, coefficients + other_offset) || l2_distance(coefficient_count, coefficients + coefficient_offset, coefficients + other_offset) > cluster_threshold)
            is_similar = false;
        }

    // If deemed similar, drop the associated voxels' coefficients.
    if (is_similar)
      for (auto i = 0; i < 2; i++)
        for (auto j = 0; j < 2; j++)
          for (auto k = 0; k < (limit_to_2d ? 1 : 2); k++)
          {
            auto other_offset = coefficient_count * (lower_layer_offset + (2 * z + k) + lower_layer_dimensions.z * (2 * y + j + lower_layer_dimensions.y * (2 * x + i)));
            for (auto c = 0; c < coefficient_count; c++)
              coefficients[other_offset + c] = 0.0;
          }
    // Else, drop this voxel's coefficients.
    else
      for (auto c = 0; c < coefficient_count; c++)
        coefficients[coefficient_offset + c] = 0.0;
  }
}

void calculate_odfs(
  cublasHandle_t     cublas            , 
  cusolverDnHandle_t cusolver          , 
  const uint3&       dimensions        , 
  const uint3&       vectors_dimensions, 
  const uint2&       histogram_bins    , 
  const unsigned     maximum_degree    , 
  const float3*      unit_vectors      , 
        float*       coefficients      , 
        std::function<void(const std::string&)> status_callback)
{
  auto start_time = std::chrono::system_clock::now();

  status_callback("Allocating and copying vectors.");
  auto voxel_count        = dimensions  .x * dimensions  .y * dimensions  .z;
  auto total_vector_count = voxel_count * vectors_dimensions.x * vectors_dimensions.y * vectors_dimensions.z;
  thrust::device_vector<float3> gpu_vectors(total_vector_count);
  auto gpu_vectors_ptr = raw_pointer_cast(&gpu_vectors[0]);
  copy_n(unit_vectors, total_vector_count, gpu_vectors.begin());
  cudaDeviceSynchronize();

  status_callback("Allocating histogram vector and magnitudes.");
  auto histogram_bin_count = histogram_bins.x * histogram_bins.y;
  thrust::device_vector<float3> histogram_vectors(histogram_bin_count);
  auto histogram_vectors_ptr = raw_pointer_cast(&histogram_vectors[0]);

  status_callback("Allocating spherical harmonics basis matrix.");
  auto coefficient_count = pli::coefficient_count(maximum_degree);
  auto matrix_size       = histogram_bin_count * coefficient_count;
  thrust::device_vector<float> basis_matrix(matrix_size, 0.0F);
  auto basis_matrix_ptr = raw_pointer_cast(&basis_matrix[0]);

  status_callback("Allocating spherical harmonics coefficients.");
  auto total_coefficient_count = voxel_count * coefficient_count;
  thrust::device_vector<float> gpu_coefficients(total_coefficient_count);
  auto gpu_coefficients_ptr = raw_pointer_cast(&gpu_coefficients[0]);

  status_callback("Generating histogram bins.");
  create_bins_kernel<<<grid_size_2d(dim3(histogram_bins.x, histogram_bins.y)), block_size_2d()>>>(
    histogram_bins       , 
    histogram_vectors_ptr);
  cudaDeviceSynchronize();

  status_callback("Calculating spherical harmonics basis matrix.");
  calculate_matrix<<<grid_size_2d(dim3(histogram_bin_count, coefficient_count)), block_size_2d()>>>(
    histogram_bin_count  , 
    coefficient_count    ,
    histogram_vectors_ptr, 
    basis_matrix_ptr     );
  cudaDeviceSynchronize();

  status_callback("Calculating SVD work buffer size.");
  int buffer_size;
  cusolverDnSgesvd_bufferSize(cusolver, histogram_bin_count, coefficient_count, &buffer_size);
  cudaDeviceSynchronize();
  auto complex_buffer_size = static_cast<float>(buffer_size);

  status_callback("Allocating SVD buffers.");
  thrust::device_vector<float> buffer (buffer_size, 0.0);
  thrust::device_vector<int>   info   (1);
  thrust::device_vector<float> U      (histogram_bin_count * histogram_bin_count);
  thrust::device_vector<float> E      (coefficient_count                        );
  thrust::device_vector<float> VT     (coefficient_count   * coefficient_count  );
  thrust::device_vector<float> UT     (histogram_bin_count * histogram_bin_count);
  thrust::device_vector<float> EI_UT  (histogram_bin_count * coefficient_count  );
  thrust::device_vector<float> V_EI_UT(histogram_bin_count * coefficient_count  );
  auto alpha       = 1.0F;
  auto beta        = 0.0F;
  auto buffer_ptr  = raw_pointer_cast(&buffer [0]);
  auto info_ptr    = raw_pointer_cast(&info   [0]);
  auto U_ptr       = raw_pointer_cast(&U      [0]);
  auto E_ptr       = raw_pointer_cast(&E      [0]);
  auto VT_ptr      = raw_pointer_cast(&VT     [0]);
  auto UT_ptr      = raw_pointer_cast(&UT     [0]);
  auto EI_UT_ptr   = raw_pointer_cast(&EI_UT  [0]);
  auto V_EI_UT_ptr = raw_pointer_cast(&V_EI_UT[0]);

  status_callback("Applying SVD.");
  cusolverDnSgesvd(
    cusolver            ,
    'A'                 ,
    'A'                 ,
    histogram_bin_count ,
    coefficient_count   ,
    basis_matrix_ptr    ,
    histogram_bin_count ,
    E_ptr               ,
    U_ptr               ,
    histogram_bin_count ,
    VT_ptr              ,
    coefficient_count   ,
    buffer_ptr          ,
    buffer_size         ,
    &complex_buffer_size,
    info_ptr            );
  cudaDeviceSynchronize();
  buffer.clear();

  status_callback("Transposing U to U^T.");
  cublasSgeam(
    cublas              ,
    CUBLAS_OP_T         ,
    CUBLAS_OP_N         ,
    histogram_bin_count ,
    histogram_bin_count ,
    &alpha              ,
    U_ptr               ,
    histogram_bin_count ,
    &beta               ,
    nullptr             ,
    histogram_bin_count ,
    UT_ptr              ,
    histogram_bin_count);
  cudaDeviceSynchronize();
  U.clear();

  status_callback("Inverting E to E^-1.");
  thrust::transform(
    E.begin(),
    E.end  (),
    E.begin(),
    [] __host__ __device__(float& entry) -> float
    {
      if (int(entry) == 0)
        return 0;
      return entry = 1.0F / entry;
    });
  cudaDeviceSynchronize();

  status_callback("Computing E^-1 * U^T.");
  cublasSdgmm(
    cublas             ,
    CUBLAS_SIDE_LEFT   ,
    coefficient_count  ,
    histogram_bin_count,
    UT_ptr             ,
    histogram_bin_count,
    E_ptr              ,
    1                  ,
    EI_UT_ptr          ,
    coefficient_count);
  cudaDeviceSynchronize();
  UT.clear();
  E .clear();

  status_callback("Computing V * E^-1 U^T.");
  cublasSgemm(
    cublas             ,
    CUBLAS_OP_T        ,
    CUBLAS_OP_N        ,
    coefficient_count  ,
    histogram_bin_count,
    coefficient_count  ,
    &alpha             ,
    VT_ptr             ,
    coefficient_count  ,
    EI_UT_ptr          ,
    coefficient_count  ,
    &beta              ,
    V_EI_UT_ptr        ,
    coefficient_count);
  cudaDeviceSynchronize();
  VT   .clear();
  EI_UT.clear();

  status_callback("Accumulating histograms and projecting via V E^-1 U^T * h. This might take a while.");
  quantify_and_project_kernel<<<dimensions, dim3(1, 1, 1)>>> (
    dimensions             ,
    vectors_dimensions     ,
    gpu_vectors_ptr        ,
    histogram_bin_count    ,
    histogram_vectors_ptr  ,
    maximum_degree         ,
    V_EI_UT_ptr            ,
    gpu_coefficients_ptr   );
  cudaDeviceSynchronize();
  gpu_vectors      .clear();
  histogram_vectors.clear();

  status_callback("Copying coefficients to CPU.");
  cudaMemcpy(coefficients, gpu_coefficients_ptr, sizeof(float) * total_coefficient_count, cudaMemcpyDeviceToHost);

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  status_callback("Cuda ODF calculation operations took " + std::to_string(elapsed_time.count()) + " seconds.");
}

void sample_odfs(
  const uint3&       dimensions       ,
  const unsigned     maximum_degree   ,
  const float*       coefficients     ,
  const uint2&       tessellations    ,
  const uint3&       vector_dimensions,
  const float        scale            ,
        float3*      points           ,
        float4*      colors           ,
        unsigned*    indices          ,
        bool         hierarchical     ,
        bool         clustering       ,
        float        cluster_threshold,
        std::function<void(const std::string&)> status_callback)
{
  auto start_time = std::chrono::system_clock::now();

  auto base_voxel_count = dimensions.x * dimensions.y * dimensions.z;
  auto dimension_count  = dimensions.z > 1 ? 3 : 2;
  auto min_dimension    = min(dimensions.x, dimensions.y);
  if (dimension_count == 3)
    min_dimension       = min(min_dimension, dimensions.z);
  auto max_layer        = int(log(min_dimension) / log(2));

  auto voxel_count = base_voxel_count;
  if (hierarchical)
    voxel_count = unsigned(base_voxel_count * 
      ((1.0 - pow(1.0 / pow(2, dimension_count), max_layer + 1)) / 
       (1.0 -     1.0 / pow(2, dimension_count))));

  auto coefficient_count  = pli::coefficient_count(maximum_degree);
  auto tessellation_count = tessellations.x * tessellations.y;
  auto point_count        = voxel_count * tessellation_count;

  status_callback("Allocating and copying the leaf spherical harmonics coefficients.");
  thrust::device_vector<float> coefficient_vectors(voxel_count * coefficient_count);
  auto coefficients_ptr = raw_pointer_cast(&coefficient_vectors[0]);

  copy_n(coefficients, base_voxel_count * coefficient_count, coefficient_vectors.begin());
  cudaDeviceSynchronize();

  auto layer_offset     = 0;
  auto layer_dimensions = dimensions;
  for (auto layer = max_layer; layer >= (hierarchical ? 0 : max_layer); layer--)
  {
    if (layer != max_layer)
    {
      status_callback("Calculating the layer " + std::to_string(int(layer)) + " coefficients.");
      sum_and_cluster_kernel<<<grid_size_3d(layer_dimensions), block_size_3d()>>>(
        layer_dimensions    ,
        layer_offset        ,
        maximum_degree      ,
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
  for (auto layer = max_layer; layer >= (hierarchical ? 0 : max_layer); layer--)
  {
    status_callback("Sampling sums of the layer " + std::to_string(int(layer)) + " coefficients.");
    sample_sums<<<grid_size_3d(layer_dimensions), block_size_3d()>>>(
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
    [] __host__ __device__(const float3& point)
    {
      return to_cartesian_coords(point);
    });
  cudaDeviceSynchronize();

  status_callback("Assigning colors.");
  thrust::transform(
    thrust::device,
    points,
    points + point_count,
    colors,
    [] __host__ __device__(const float3& point)
    {
      return make_float4(abs(point.x), abs(point.z), abs(point.y), 1.0);
    });
  cudaDeviceSynchronize();

  status_callback("Translating and scaling the points.");
  layer_offset     = 0;
  layer_dimensions = dimensions;
  for (auto layer = max_layer; layer >= (hierarchical ? 0 : max_layer); layer--)
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
      vector_dimensions.x * unsigned(pow(2, max_layer - layer)),
      vector_dimensions.y * unsigned(pow(2, max_layer - layer)),
      vector_dimensions.z * unsigned(pow(2, max_layer - layer)) };
    float3 layer_position {
      (layer_vectors_size.x - 1.0F) * 0.5F,
      (layer_vectors_size.y - 1.0F) * 0.5F,
      dimension_count == 3 ? (layer_vectors_size.z - 1) * 0.5F : 0.0F };
    float3 layer_spacing {
      layer_vectors_size.x,
      layer_vectors_size.y,
      dimension_count == 3 ? layer_vectors_size.z : 1.0F };
    float  min_spacing = min(layer_spacing.x, layer_spacing.y);
    if(dimension_count == 3)
      min_spacing = min(min_spacing, layer_spacing.z);
  
    auto layer_scale = scale * min_spacing  * 0.5F;
  
    thrust::transform(
      thrust::device,
      points + layer_point_offset,
      points + layer_point_offset + layer_point_count,
      points + layer_point_offset,
      [=] __host__ __device__(const float3& point)
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

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  status_callback("Cuda ODF sampling operations took " + std::to_string(elapsed_time.count()) + " seconds.");
}

void extract_peaks(
  const uint3&   dimensions    ,
  const unsigned maximum_degree,
  const float*   coefficients  ,
  const uint2&   tessellations , 
  const unsigned maxima_count  ,
        float3*  maxima        ,
        std::function<void(const std::string&)> status_callback)
{
  auto start_time = std::chrono::system_clock::now();

  auto voxel_count       = dimensions.x * dimensions.y * dimensions.z;
  auto coefficient_count = pli::coefficient_count(maximum_degree);

  status_callback("Allocating and copying the spherical harmonics coefficients.");
  thrust::device_vector<float> coefficient_vectors(voxel_count * coefficient_count);
  auto coefficients_ptr = raw_pointer_cast(&coefficient_vectors[0]);
  copy_n(coefficients, coefficient_vectors.size(), coefficient_vectors.begin());
  cudaDeviceSynchronize();

  status_callback("Allocating the maxima vectors.");
  thrust::device_vector<float3> maxima_vectors(voxel_count * maxima_count);
  auto maxima_ptr = raw_pointer_cast(&maxima_vectors[0]);

  status_callback("Extracting maxima.");
  extract_maxima<<<grid_size_3d(dimensions), block_size_3d()>>>(
    dimensions       ,
    coefficient_count,
    coefficients_ptr ,
    tessellations    ,
    maxima_count     ,
    maxima_ptr       );
  cudaDeviceSynchronize();

  status_callback("Converting the maxima to Cartesian coordinates.");
  thrust::transform(
    thrust::device,
    maxima_vectors.begin(),
    maxima_vectors.end  (),
    maxima_vectors.begin(),
    [] __host__ __device__(const float3& point)
    {
      return to_cartesian_coords(point);
    });
  cudaDeviceSynchronize();

  status_callback("Copying maxima to CPU.");
  cudaMemcpy(maxima, maxima_ptr, sizeof(float3) * maxima_vectors.size(), cudaMemcpyDeviceToHost);

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  status_callback("Cuda ODF sampling operations took " + std::to_string(elapsed_time.count()) + " seconds.");
}
}
