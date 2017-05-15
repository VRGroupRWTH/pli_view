#ifndef CUSH_LEGENDRE_H_
#define CUSH_LEGENDRE_H_

#include <math.h>

#include <sh/decorators.h>
#include <sh/factorial.h>

namespace cush
{
// Based on the recurrence relations defined in page 10 of
// "Spherical Harmonic Lighting: The Gritty Details" by Robin Green.
template<typename precision>
INLINE COMMON precision associated_legendre(const int l, const int m, const precision& x)
{
  precision p_mm(1.0);
  if (l > 0)
    p_mm = (m % 2 == 0 ? 1 : -1) * double_factorial<precision>(fmax(2.0F * m - 1.0F, 0.0F)) * pow(1 - x * x, m / 2.0);
  if (l == m)
    return p_mm;
  
  precision p_m1m = x * (2 * m + 1) * p_mm;
  if (l == m + 1) 
    return p_m1m;

  for (auto n = m + 2; n <= l; n++)
  {
    precision p_lm = (x * (2 * n - 1) * p_m1m - (n + m - 1) * p_mm) / (n - m);
    p_mm  = p_m1m;
    p_m1m = p_lm;
  }
  return p_m1m;
}
}

#endif
