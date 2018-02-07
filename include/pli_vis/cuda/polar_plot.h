#ifndef PLI_VIS_POLAR_PLOT_H_
#define PLI_VIS_POLAR_PLOT_H_

#include <vector>

#include <vector_types.h>

namespace pli
{
std::array<std::vector<float3>, 2> calculate(
  const std::vector<float3>& vectors           ,
  const uint2&               vectors_dimensions,
  const unsigned             superpixel_size   ,
  const unsigned             angular_partitions,
  const bool                 symmetric         );
}

#endif