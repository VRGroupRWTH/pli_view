#ifndef PLI_VIS_CLEBSCH_GORDAN_H_
#define PLI_VIS_CLEBSCH_GORDAN_H_

#include <host_defines.h>
#include <math.h>

#include <pli_vis/cuda/sh/wigner.h>

namespace pli
{
// Based on "Wigner 3j-Symbol." of Eric Weisstein at http://mathworld.wolfram.com/Wigner3j-Symbol.html
template<typename precision>
__host__ __device__ precision clebsch_gordan(
  unsigned int l1, unsigned int l2, unsigned int l3,
  unsigned int m1, unsigned int m2, unsigned int m3)
{
  return pow (precision(-1.0) , m3 + l1 - l2) *   
         sqrt(precision( 2.0) * l3 + precision(1.0)) *
         wigner_3j<precision>(precision(2.0) * l1, precision(2.0) * l2,  precision(2.0) * l3, 
                              precision(2.0) * m1, precision(2.0) * m2, -precision(2.0) * m3);
}
}

#endif
