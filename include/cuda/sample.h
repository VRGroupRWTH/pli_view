#ifndef PLI_VIS_SAMPLER_H_
#define PLI_VIS_SAMPLER_H_

#include <vector_types.h>

namespace pli
{
void sample_sphere(
  const uint2&    tessellations ,
        float3*   points        ,
        unsigned* indices       );

void sample_sums(
  const uint3&    dimensions    , 
  const unsigned  maximum_degree,
  const uint2&    tessellations , 
  const float*    coefficients  , 
        float3*   points        , 
        unsigned* indices       );
}

#endif