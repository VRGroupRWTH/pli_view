#include <pli_vis/cuda/zernike/launch.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <pli_vis/cuda/utility/vector_ops.h>
#include <pli_vis/cuda/zernike/disk.h>
#include <pli_vis/cuda/zernike/zernike.h>

namespace zer
{
__host__ thrust::device_vector<float> pseudoinverse(const uint2& size, thrust::device_vector<float>& data)
{
  cublasHandle_t     cublas  ;
  cusolverDnHandle_t cusolver;
  cusolverDnCreate(&cusolver);
  cublasCreate    (&cublas  );

  int buffer_size;
  cusolverDnSgesvd_bufferSize(cusolver, size.x, size.y, &buffer_size);
  cudaDeviceSynchronize      ();
  auto complex_buffer_size = static_cast<float>(buffer_size);
  
  thrust::device_vector<float> buffer (buffer_size, 0.0);
  thrust::device_vector<int>   info   (1);
  thrust::device_vector<float> u      (size.x * size.x);
  thrust::device_vector<float> e      (size.y);
  thrust::device_vector<float> vt     (size.y * size.y);
  thrust::device_vector<float> ut     (size.x * size.x);
  thrust::device_vector<float> ei_ut  (size.x * size.y);
  thrust::device_vector<float> v_ei_ut(size.x * size.y);
  auto alpha = 1.0F;
  auto beta  = 0.0F;
  
  cusolverDnSgesvd(
    cusolver            ,
    'A'                 ,
    'A'                 ,
    size.x              ,
    size.y              ,
    data.data().get()   ,
    size.x              ,
    e.data().get()      ,
    u.data().get()      ,
    size.x              ,
    vt.data().get()     ,
    size.y              ,
    buffer.data().get() ,
    buffer_size         ,
    &complex_buffer_size,
    info.data().get()   );
  cudaDeviceSynchronize();
  buffer.clear();

  cublasSgeam(
    cublas         ,
    CUBLAS_OP_T    ,
    CUBLAS_OP_N    ,
    size.x         ,
    size.x         ,
    &alpha         ,
    u.data().get() ,
    size.x         ,
    &beta          ,
    nullptr        ,
    size.x         ,
    ut.data().get(),
    size.x         );
  cudaDeviceSynchronize();
  u.clear();
  
  thrust::transform(
    e.begin(),
    e.end  (),
    e.begin(),
    [] __host__ __device__(float& entry) -> float
    {
      if (int(entry) == 0)
        return 0;
      return entry = 1.0F / entry;
    });
  cudaDeviceSynchronize();

  cublasSdgmm(
    cublas            ,
    CUBLAS_SIDE_LEFT  ,
    size.y            ,
    size.x            ,
    ut.data().get()   ,
    size.x            ,
    e.data().get()    ,
    1                 ,
    ei_ut.data().get(),
    size.y            );
  cudaDeviceSynchronize();
  ut.clear();
  e .clear();

  cublasSgemm(
    cublas              ,
    CUBLAS_OP_T         ,
    CUBLAS_OP_N         ,
    size.y              ,
    size.x              ,            
    size.y              ,
    &alpha              ,
    vt.data().get()     ,
    size.y              ,
    ei_ut.data().get()  ,
    size.y              ,
    &beta               ,
    v_ei_ut.data().get(),
    size.y              );
  cudaDeviceSynchronize();
  vt   .clear();
  ei_ut.clear();

  cusolverDnDestroy(cusolver);
  cublasDestroy    (cublas  );

  return v_ei_ut;
}

__global__ void accumulate(
  const uint2    vectors_size         ,
  const float3*  vectors              ,
  const uint2    disk_partitions      ,
  const float2*  disk_samples         ,
  const uint2    superpixel_size      ,
  const uint2    superpixel_dimensions,
        float*   intermediates        )
{
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= vectors_size.x || y >= vectors_size.y)
    return;

  const auto& vector = vectors[y + vectors_size.y * x];

  const auto superpixel_x        = x / superpixel_size.x;
  const auto superpixel_y        = y / superpixel_size.y;
  const auto superpixel_index    = superpixel_y + superpixel_dimensions.y * superpixel_x;
  const auto sample_count        = disk_partitions.x * disk_partitions.y;
  const auto intermediate_offset = sample_count * superpixel_index;

  auto min_distance = 2.0F;
  auto sample_index = 0;
  for(auto i = 0; i < sample_count; i++)
  {
    const auto temp_distance = sqrt(pow(cos(vector.z), 2) + pow(disk_samples[i].x, 2) - 2.0F * cos(vector.z) * disk_samples[i].x * cos(vector.y - disk_samples[i].y));
    if (temp_distance < min_distance)
    {
      min_distance = temp_distance;
      sample_index = i;
    }
  }

  atomicAdd(&intermediates[intermediate_offset + sample_index], 1.0F);
}

__global__ void project(
  const uint2    dimensions           ,
  const float*   basis_matrix         ,
  const unsigned sample_count         ,
  const float*   intermediates        ,
  const unsigned coefficient_count    ,
        float*   coefficients         )
{
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dimensions.x || y >= dimensions.y)
    return;

  const auto linear_index        = y + dimensions.y * x;
  const auto intermediate_offset = sample_count      * linear_index;
  const auto coefficient_offset  = coefficient_count * linear_index;

  cublasHandle_t cublas;
  cublasCreate(&cublas);
  auto alpha = 1.0F;
  auto beta  = 0.0F;
  cublasSgemv(
    cublas                             ,
    CUBLAS_OP_N                        ,
    coefficient_count                  ,
    sample_count                       ,
    &alpha                             ,
    basis_matrix                       ,
    coefficient_count                  ,
    &intermediates[intermediate_offset],
    1                                  ,
    &beta                              ,
    &coefficients [coefficient_offset ],
    1                                  );
  cublasDestroy(cublas);
}

thrust::device_vector<float> launch(
  const thrust::device_vector<float3>& vectors        ,
  const uint2&                         vectors_size   ,
  const uint2&                         superpixel_size,
  const uint2&                         disk_partitions,
  const unsigned                       maximum_degree )
{
  const auto superpixel_dimensions = uint2{vectors_size.x / superpixel_size.x, vectors_size.y / superpixel_size.y};
  const auto superpixel_count      = superpixel_dimensions.x * superpixel_dimensions.y;
  const auto sample_count          = disk_partitions.x * disk_partitions.y;
  const auto coefficient_count     = expansion_size(maximum_degree);

  // Sample a unit disk.
  thrust::device_vector<float2> disk_samples(sample_count);
  sample_disk<<<grid_size_2d(dim3(disk_partitions.x, disk_partitions.y)), block_size_2d()>>>(
    disk_partitions           , 
    disk_samples.data().get());

  // Compute Zernike basis for the samples.
  thrust::device_vector<float> basis_matrix(sample_count * coefficient_count);
  compute_basis<<<grid_size_2d(dim3(sample_count, coefficient_count)), block_size_2d()>>>(
    sample_count              , 
    disk_samples.data().get() , 
    coefficient_count         ,
    basis_matrix.data().get());

  // Compute the inverse of the basis matrix.
  auto inverse_basis_matrix = pseudoinverse({sample_count, coefficient_count}, basis_matrix);
  
  // Accumulate vectors into superpixels.
  thrust::device_vector<float> intermediates(superpixel_count * sample_count);
  accumulate<<<grid_size_2d(dim3(vectors_size.x, vectors_size.y)), block_size_2d()>>> (
    vectors_size              ,
    vectors      .data().get(),
    disk_partitions           ,
    disk_samples .data().get(),
    superpixel_size           ,
    superpixel_dimensions     ,
    intermediates.data().get());

  // Project superpixels to the Zernike basis.
  thrust::device_vector<float> coefficients(superpixel_count * coefficient_count);
  project<<<grid_size_2d(dim3(superpixel_dimensions.x, superpixel_dimensions.y)), block_size_2d()>>> (
    superpixel_dimensions            ,
    inverse_basis_matrix.data().get(),
    sample_count                     ,
    intermediates.data().get()       ,
    coefficient_count                ,
    coefficients.data().get())       ;

  return coefficients;
}
}
