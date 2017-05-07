#include /* implements */ <cuda/odf_field.h>

#include <chrono>
#include <string>

#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <sh/cush.h>
#include <sh/vector_ops.h>

#include <cuda/launch.h>
#include <cuda/spherical_histogram.h>

namespace pli
{
void calculate_odfs(
  const uint3&   dimensions    , 
  const uint3&   vectors_size  , 
  const uint2&   histogram_bins,
  const unsigned maximum_degree, 
  const float*   directions    , 
  const float*   inclinations  , 
        float*   coefficients  )
{
  auto total_start = std::chrono::system_clock::now();

  std::cout << "Resetting GPU." << std::endl;
  cudaDeviceReset();

  std::cout << "Allocating and copying vectors." << std::endl;
  auto voxel_count        = dimensions  .x * dimensions  .y * dimensions  .z;
  auto vector_count       = vectors_size.x * vectors_size.y * vectors_size.z;
  auto total_vector_count = voxel_count * vector_count;
  thrust::device_vector<float> longitudes(total_vector_count);
  thrust::device_vector<float> latitudes (total_vector_count);
  copy_n(directions  , total_vector_count, longitudes.begin());
  copy_n(inclinations, total_vector_count, latitudes .begin());
  auto longitudes_ptr = raw_pointer_cast(&longitudes[0]);
  auto latitudes_ptr  = raw_pointer_cast(&latitudes [0]);
  transform(longitudes.begin(), longitudes.end(), longitudes.begin(), (90.0 + thrust::placeholders::_1) * M_PI / 180.0);
  transform(latitudes .begin(), latitudes .end(), latitudes .begin(), (90.0 - thrust::placeholders::_1) * M_PI / 180.0);
  cudaDeviceSynchronize();

  std::cout << "Allocating histogram vector and magnitudes." << std::endl;
  auto histogram_bin_count = histogram_bins.x * histogram_bins.y;
  thrust::device_vector<float3> histogram_vectors   (histogram_bin_count);
  thrust::device_vector<float > histogram_magnitudes(histogram_bin_count);
  auto histogram_vectors_ptr    = raw_pointer_cast(&histogram_vectors   [0]);
  auto histogram_magnitudes_ptr = raw_pointer_cast(&histogram_magnitudes[0]);

  std::cout << "Allocating spherical harmonics basis matrix." << std::endl;
  auto coefficient_count = cush::coefficient_count(maximum_degree);
  auto matrix_size       = histogram_bin_count * coefficient_count;
  thrust::device_vector<float> basis_matrix(matrix_size, 0.0F);
  auto basis_matrix_ptr = raw_pointer_cast(&basis_matrix[0]);

  std::cout << "Allocating spherical harmonics coefficients." << std::endl;
  auto total_coefficient_count = voxel_count * coefficient_count;
  thrust::device_vector<float> coefficient_vectors(total_coefficient_count);
  auto coefficient_vectors_ptr = raw_pointer_cast(&coefficient_vectors[0]);

  std::cout << "Generating histogram bins." << std::endl;
  create_bins<<<grid_size_2d(dim3(histogram_bins.x, histogram_bins.y)), block_size_2d()>>>(
    histogram_bins       , 
    histogram_vectors_ptr);
  cudaDeviceSynchronize();

  std::cout << "Calculating spherical harmonics basis matrix." << std::endl;
  cush::calculate_matrix<<<grid_size_2d(dim3(histogram_bin_count, coefficient_count)), block_size_2d()>>>(
    histogram_bin_count  , 
    coefficient_count    ,
    histogram_vectors_ptr, 
    basis_matrix_ptr     );
  cudaDeviceSynchronize();

  std::cout << "Initializing cusolver." << std::endl;
  cusolverDnHandle_t cusolver;  
  cusolverDnCreate(&cusolver);

  std::cout << "Initializing cublas." << std::endl;
  cublasHandle_t cublas;
  cublasCreate(&cublas);

  std::cout << "Calculating SVD work buffer size." << std::endl;
  int buffer_size;
  cusolverDnSgesvd_bufferSize(cusolver, histogram_bin_count, coefficient_count, &buffer_size);
  cudaDeviceSynchronize();
  float complex_buffer_size = buffer_size;

  std::cout << "Allocating SVD buffers." << std::endl;
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

  std::cout << "Applying SVD." << std::endl;
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
    info_ptr);
  cudaDeviceSynchronize();

  std::cout << "Transposing U to U^T." << std::endl;
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

  std::cout << "Inverting E to E^-1." << std::endl;
  thrust::transform(
    E.begin(),
    E.end  (),
    E.begin(),
    [] COMMON (float& entry) -> float
    {
      if (int(entry) == 0)
        return 0;
      return entry = 1.0 / entry;
    });
  cudaDeviceSynchronize();

  std::cout << "Computing E^-1 * U^T." << std::endl;
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

  std::cout << "Computing V * E^-1 U^T." << std::endl;
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

  std::cout << "Accumulating histograms and projecting via V E^-1 U^T * h." << std::endl;
  for (auto x = 0; x < dimensions.x; x++)
  {
    for (auto y = 0; y < dimensions.y; y++)
    {
      for (auto z = 0; z < dimensions.z; z++)
      {
        auto volume_index        = z + dimensions.z * (y + dimensions.y * x);
        auto coefficients_offset = volume_index * coefficient_count;

        fill(histogram_magnitudes.begin(), histogram_magnitudes.end(), 0.0F);

        uint3 offset {
          vectors_size.x * x,
          vectors_size.y * y,
          vectors_size.z * z};
        uint3 size  {
          vectors_size.x * dimensions.x,
          vectors_size.y * dimensions.y,
          vectors_size.z * dimensions.z};

        accumulate<<<grid_size_3d(vectors_size), block_size_3d()>>>(
          vectors_size         ,
          offset               ,
          size                 ,
          longitudes_ptr       , 
          latitudes_ptr        , 
          histogram_bins       , 
          histogram_vectors_ptr, 
          histogram_magnitudes_ptr);

        cublasSgemv(
          cublas                                        ,
          CUBLAS_OP_N                                   ,
          coefficient_count                             ,
          histogram_bin_count                           ,
          &alpha                                        ,
          V_EI_UT_ptr                                   ,
          coefficient_count                             ,
          histogram_magnitudes_ptr                      , 
          1                                             ,
          &beta                                         ,
          coefficient_vectors_ptr  + coefficients_offset,
          1                                             );

        cudaDeviceSynchronize();

        std::cout << volume_index + 1 << "/" << voxel_count << "." << std::endl;
      }
    }
  }

  std::cout << "Copying coefficients to CPU." << std::endl;
  cudaMemcpy(coefficients, coefficient_vectors_ptr, sizeof(float) * total_coefficient_count, cudaMemcpyDeviceToHost);
  
  std::cout << "Destroying cublas." << std::endl;
  cublasDestroy    (cublas  );

  std::cout << "Destroying cusolver." << std::endl;
  cusolverDnDestroy(cusolver);

  auto total_end = std::chrono::system_clock::now();
  std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
  std::cout << "Total elapsed time: " << total_elapsed_seconds.count() << "s." << std::endl;
}

void sample_odfs(
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
      sample_odf_layer<<<grid_size_3d(layer_dimensions), block_size_3d()>>>(
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
