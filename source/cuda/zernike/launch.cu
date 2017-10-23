#include <pli_vis/cuda/zernike/launch.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <pli_vis/cuda/zernike/disk.h>
#include <pli_vis/cuda/zernike/zernike.h>
#include "pli_vis/cuda/sh/spherical_harmonics.h"

namespace zer
{
thrust::device_vector<float> pseudoinverse(
  const uint2&                        size, 
  const thrust::device_vector<float>& data)
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
  auto alpha       = 1.0F;
  auto beta        = 0.0F;
  auto buffer_ptr  = raw_pointer_cast(&buffer [0]);
  auto info_ptr    = raw_pointer_cast(&info   [0]);
  auto u_ptr       = raw_pointer_cast(&u      [0]);
  auto e_ptr       = raw_pointer_cast(&e      [0]);
  auto vt_ptr      = raw_pointer_cast(&vt     [0]);
  auto ut_ptr      = raw_pointer_cast(&ut     [0]);
  auto ei_ut_ptr   = raw_pointer_cast(&ei_ut  [0]);
  auto v_ei_ut_ptr = raw_pointer_cast(&v_ei_ut[0]);
  
  // TODO: Run cusolverDnSgesvd UEV, cublasSgeam U, thrust::transform E, cublasSdgmm V, cublasSgemm VEU.

  cusolverDnDestroy(cusolver);
  cublasDestroy    (cublas  );
}

thrust::device_vector<float> launch(
  const thrust::device_vector<float3>& vectors        ,
  const uint2&                         superpixel_size,
  const uint2&                         disk_partitions,
  const unsigned                       maximum_degree ,
  const bool                           symmetric      )
{
  const auto superpixel_count  = vectors.size() / (superpixel_size.x * superpixel_size.y);
  const auto sample_count      = disk_partitions.x * disk_partitions.y;
  const auto coefficient_count = expansion_size(maximum_degree);

  thrust::device_vector<float> coefficients(superpixel_count * coefficient_count);

  // TODO: Compute the unprojected form of each superpixel.
  
  // Sample a unit disk.
  thrust::device_vector<float2> disk_samples(sample_count);
  sample_disk<<<grid_size_2d(dim3(superpixel_size.x, superpixel_size.y)), block_size_2d()>>>(
    disk_partitions           , 
    disk_samples.data().get());

  // Compute Zernike basis for the unit disk.
  thrust::device_vector<float> basis_matrix(sample_count * coefficient_count);
  compute_basis<<<grid_size_2d(dim3(sample_count, coefficient_count)), block_size_2d()>>>(
    sample_count              , 
    disk_samples.data().get() , 
    coefficient_count            , 
    basis_matrix.data().get());

  // Compute the inverse of the basis matrix.
  auto inverse = pseudoinverse({sample_count, coefficient_count}, basis_matrix);

  // TODO: Multiply the inverse matrix with the unprojected form to obtain the coefficients.

  return coefficients;
}
}
