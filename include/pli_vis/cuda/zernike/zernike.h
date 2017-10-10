#ifndef ZERNIKE_H_
#define ZERNIKE_H_

#include <math.h>

#include <host_defines.h>
#include <vector_types.h>

// References:
// - Lakshminarayanan & Fleck, Zernike Polynomials: A Guide, Journal of Modern Optics 2011.7.
// Notes:
// - All angles are in radians.
// - Radius is restricted to the unit circle.
// - Theta is measured clockwise from the vertical axis.
namespace zer
{
__forceinline__ __host__ __device__ unsigned maximum_degree(const unsigned& count)
{
  return round(sqrt(2 * count) - 1);
}
__forceinline__ __host__ __device__ unsigned expansion_size(const unsigned& max_n)
{
  return (max_n + 1) * (max_n + 2) / 2;
}
__forceinline__ __host__ __device__ unsigned linear_index  (const int2&     nm   )
{
  return (nm.x * (nm.x + 2) + nm.y) / 2;
}
__forceinline__ __host__ __device__ int2     quantum_index (const unsigned& i    )
{
  int2 nm;
  nm.x = ceil((-3 + sqrt(9 + 8 * i)) / 2);
  nm.y = 2 * i - nm.x * (nm.x + 2);
  return nm;
}

template<typename precision>
__host__ __device__ precision factorial(unsigned n)
{
  precision out(1);
  for (auto i = 2; i <= n; i++) 
    out *= i;
  return out;
}
template<typename precision>
__host__ __device__ precision mode     (const uint2&   index, const precision& rho)
{
  precision sum(0);
  for(unsigned i = 0; i < (index.x - index.y) / 2; i++)
    sum += pow(rho, index.x - 2 * i) * 
     (pow(-1, i) * factorial<precision>(index.x - i)) / 
     (factorial<precision>(i) * 
      factorial<precision>(0.5 * (index.x + index.y) - i) * 
      factorial<precision>(0.5 * (index.x - index.y) - i));
  return sum;
}

template<typename precision>
__host__ __device__ precision evaluate(const int2&    index, const precision& rho, const precision& theta)
{
  return mode(index, rho) * (index.y >= 0 ? cos(index.y * theta) : sin(index.y * theta));
}
template<typename precision>
__host__ __device__ precision evaluate(const unsigned index, const precision& rho, const precision& theta)
{
  return evaluate(quantum_index(index), rho, theta);
}

// TODO: Calculate basis matrix kernel(s).
// TODO: Sample sum kernel(s).
}

#endif
