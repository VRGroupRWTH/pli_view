#ifndef CUPT_TRACER_H_
#define CUPT_TRACER_H_

#include <cstddef>
#include <vector>

#include <vector_types.h>

namespace cupt 
{
std::vector<std::vector<float4>> trace(
  const std::size_t          iteration_count,
  const float                step_size      ,
  const uint3                data_dimensions,
  const float4               data_spacing   ,
  const std::vector<float4>& data           ,
  const std::vector<float4>& seeds          );
}

#endif
