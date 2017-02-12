#ifndef PLI_VIS_SAMPLER_H_
#define PLI_VIS_SAMPLER_H_

#include <vector_types.h>

namespace pli
{
void sample(
  const uint3&   dimensions    , 
  const uint2&   tessellations , 
  const unsigned maximum_degree,
  const float*   coefficients  , 
        float3*  points        , 
        float*   indices       );
}

#endif