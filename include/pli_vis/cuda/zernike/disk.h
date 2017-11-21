#ifndef DISK_H_
#define DISK_H_

#define _USE_MATH_DEFINES

#include <math.h>

#include <device_launch_parameters.h>
#include <host_defines.h>
#include <vector_types.h>

// References:
// - Weisstein, Disk Point Picking, MathWorld - A Wolfram Web Resource.
namespace zer
{
// This kernel requires a dimensions.x x dimensions.y 2D grid.
template<typename precision>
__global__ void sample_disk(uint2 dimensions, precision* samples, bool uniform)
{
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dimensions.x || y >= dimensions.y)
    return;

  const auto rho   = uniform ? sqrtf(static_cast<float>(x) / dimensions.x) : static_cast<float>(x) / dimensions.x;
  const auto theta = 2.0f * M_PI * y / dimensions.y; 

  const auto sample_index = y + dimensions.y * x;
  samples[sample_index].x = rho * cos(theta);
  samples[sample_index].y = rho * sin(theta);
}
}

#endif