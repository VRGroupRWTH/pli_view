#ifndef CUSH_FACTORIAL_H_
#define CUSH_FACTORIAL_H_

#include <math.h>

#include <cuda/sh/decorators.h>

namespace cush
{
template<typename precision = double>
INLINE COMMON precision factorial          (unsigned int n)
{
  precision out(1.0);
  for (auto i = 2; i <= n; i++)
    out *= i;
  return out;
}
template<typename precision = double>
INLINE COMMON precision ln_factorial       (unsigned int n)
{
  return log(factorial<precision>(n));
}

template<typename precision = double>
INLINE COMMON precision double_factorial   (unsigned int n)
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
INLINE COMMON precision ln_double_factorial(unsigned int n)
{
  return log(double_factorial<precision>(n));
}
}

#endif
