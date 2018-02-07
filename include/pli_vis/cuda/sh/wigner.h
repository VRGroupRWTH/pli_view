#ifndef PLI_VIS_WIGNER_H_
#define PLI_VIS_WIGNER_H_

#include <host_defines.h>
#include <math.h>

#include <pli_vis/cuda/sh/choose.h>

namespace pli
{
// Based on GNU Scientific Library's implementation. Input is in half-integer units!
template<typename precision>
__host__ __device__ precision wigner_3j(int two_l1, int two_l2, int two_l3,
                                        int two_m1, int two_m2, int two_m3)
{
  if (two_l1 < 0                    ||
      two_l2 < 0                    ||
      two_l3 < 0                    ||
      two_l2 < abs(two_l1 - two_l3) || 
      two_l2 > two_l1 + two_l3      || 
      two_l1 + two_l2 + two_l3 & 1  || 
      (abs(two_m1) > two_l1         || 
       abs(two_m2) > two_l2         || 
       abs(two_m3) > two_l3         || 
      two_l1 + two_m1 & 1           || 
      two_l2 + two_m2 & 1           || 
      two_l3 + two_m3 & 1           || 
      two_m1 + two_m2 + two_m3      != 0))
    return precision(0);

  // Special case for (ja jb jc 0 0 0) = 0 when ja + jb + jc is odd.
  if (two_m1 == 0 && two_m2 == 0 && two_m3 == 0 && (two_l1 + two_l2 + two_l3) % 4 == 2)
    return precision(0);

  auto lc1   = (-two_l1 + two_l2 + two_l3) / 2;
  auto lc2   = ( two_l1 - two_l2 + two_l3) / 2;
  auto lc3   = ( two_l1 + two_l2 - two_l3) / 2;
  auto lmm1  = ( two_l1 - two_m1) / 2;
  auto lmm2  = ( two_l2 - two_m2) / 2;
  auto lmm3  = ( two_l3 - two_m3) / 2;
  auto lpm1  = ( two_l1 + two_m1) / 2;
  auto lpm2  = ( two_l2 + two_m2) / 2;
  auto lpm3  = ( two_l3 + two_m3) / 2;
  auto lsum  = ( two_l1 + two_l2 + two_l3) / 2;
  int  kmin  = fmax(fmax(0.0f                   , lpm2 - lmm3), lmm1 - lpm3);
  int  kmax  = fmin(fmin(static_cast<float>(lc3), lmm1       ), lpm2       );
  auto sign  = kmin - lpm1 + lmm2 & 1 ? -1 : 1;

  auto bcn1  = ln_choose(two_l1  , lc3 );
  auto bcn2  = ln_choose(two_l2  , lc3 );
  auto bcd1  = ln_choose(lsum + 1, lc3 );
  auto bcd2  = ln_choose(two_l1  , lmm1);
  auto bcd3  = ln_choose(two_l2  , lmm2);
  auto bcd4  = ln_choose(two_l3  , lpm3);
  auto lnorm = 0.5 * (bcn1 + bcn2 - bcd1 - bcd2 - bcd3 - bcd4 - log(two_l3 + 1.0));

  precision sum_pos(0), sum_neg(0);
  for (auto k = kmin; k <= kmax; k++)
  {
    auto bc1  = ln_choose(lc3,        k);
    auto bc2  = ln_choose(lc2, lmm1 - k);
    auto bc3  = ln_choose(lc1, lpm2 - k);
    auto term = exp(bc1 + bc2 + bc3 + lnorm);
    sign < 0 ? (sum_neg += term) : (sum_pos += term);
    sign = -sign;
  }
  return sum_pos - sum_neg;
}
}

#endif
