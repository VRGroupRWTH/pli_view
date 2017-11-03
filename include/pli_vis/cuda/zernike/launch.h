#ifndef ZERNIKE_LAUNCH_H_
#define ZERNIKE_LAUNCH_H_

#include <thrust/device_vector.h>
#include <vector_types.h>

namespace zer
{
thrust::device_vector<float> launch(
  const thrust::device_vector<float3>& vectors                       ,
  const uint2&                         vectors_size                  ,
  const uint2&                         superpixel_size               ,
  const uint2&                         disk_partitions = {100u, 360u},
  const unsigned                       maximum_degree  = 10          );
  
std::vector<float> launch(
  const std::vector<float3>&          vectors                       ,
  const uint2&                         vectors_size                  ,
  const uint2&                         superpixel_size               ,
  const uint2&                         disk_partitions = {100u, 360u},
  const unsigned                       maximum_degree  = 10          );
}

#endif
