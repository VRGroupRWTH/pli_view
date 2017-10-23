#ifndef ZERNIKE_LAUNCH_H_
#define ZERNIKE_LAUNCH_H_

#include <thrust/device_vector.h>
#include <vector_types.h>

namespace zer
{
thrust::device_vector<float> pseudoinverse(
  const uint2&                         size,
  const thrust::device_vector<float>&  data);

thrust::device_vector<float> launch(
  const thrust::device_vector<float3>& vectors                       ,
  const uint2&                         superpixel_size               ,
  const uint2&                         disk_partitions = {100u, 360u},
  const unsigned                       maximum_degree  = 6           ,
  const bool                           symmetric       = false       );
}

#endif
