#ifndef CUSH_CLEBSCH_GORDAN_H_
#define CUSH_CLEBSCH_GORDAN_H_

#include <math.h>

#include <sh/decorators.h>
#include <sh/wigner.h>

namespace cush
{
// Based on "Wigner 3j-Symbol." of Eric Weisstein at http://mathworld.wolfram.com/Wigner3j-Symbol.html
template<typename precision>
INLINE COMMON precision clebsch_gordan(
  unsigned int l1, unsigned int l2, unsigned int l3,
  unsigned int m1, unsigned int m2, unsigned int m3)
{
  return pow (precision(-1.0), m3 + l1 - l2) * 
         sqrt(2.0 * l3 + 1.0) * 
         wigner_3j<precision>(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * m2, -2 * m3);
}
}

#endif
