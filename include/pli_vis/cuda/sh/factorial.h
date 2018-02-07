#ifndef PLI_VIS_FACTORIAL_H_
#define PLI_VIS_FACTORIAL_H_

#include <host_defines.h>
#include <math.h>

namespace pli
{
template<typename precision = double>
__host__ __device__ precision factorial          (unsigned int n)
{
  precision out(1.0);
  for (auto i = 2; i <= n; i++)
    out *= i;
  return out;
}
template<typename precision = double>
__host__ __device__ precision ln_factorial       (unsigned int n)
{
  return log(factorial<precision>(n));
}

template<typename precision = double>
__host__ __device__ precision double_factorial   (unsigned int n)
{
  precision out(1.0);
  while (n > 1)
  {
    out *= n;
    n   -= 2;
  }
  return out;
}
template<typename precision = double>
__host__ __device__ precision ln_double_factorial(unsigned int n)
{
  return log(double_factorial<precision>(n));
}
}

#endif
