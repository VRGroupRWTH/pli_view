#ifndef CUSH_CONFIG_H_
#define CUSH_CONFIG_H_

#ifdef __CUDACC__

  #define GLOBAL __global__
  #define HOST   __host__
  #define DEVICE __device__
  #define SHARED __shared__
  #define INLINE __forceinline__ inline
  #define CONSTANT  __constant__ const

#else

  #define GLOBAL
  #define HOST
  #define DEVICE
  #define SHARED
  #define INLINE inline
  #define CONSTANT  const

#endif

#define COMMON HOST DEVICE

#endif
