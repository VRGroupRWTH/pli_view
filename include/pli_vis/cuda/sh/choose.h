#ifndef PLI_VIS_CHOOSE_H_
#define PLI_VIS_CHOOSE_H_

#include <host_defines.h>

#include <pli_vis/cuda/sh/factorial.h>

namespace pli
{
// Based on GNU Scientific Library's implementation.
template<typename precision = double>
__host__ __device__ precision choose   (unsigned int n, unsigned int m)
{
  return factorial<precision>(n) / (factorial<precision>(m) * factorial<precision>(n - m));
}
template<typename precision = double>
__host__ __device__ precision ln_choose(unsigned int n, unsigned int m)
{
  if (m == n || m == 0)
    return precision(0);

  if (m * 2 > n)
    m = n - m;

  return ln_factorial<precision>(n) - ln_factorial<precision>(m) - ln_factorial<precision>(n - m);
}
}

#endif
