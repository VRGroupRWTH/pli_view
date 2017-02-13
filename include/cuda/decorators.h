#ifndef PLI_ODF_DECORATORS_H_
#define PLI_ODF_DECORATORS_H_

#ifdef __CUDACC__

  #define GLOBAL __global__
  #define HOST   __host__
  #define DEVICE __device__
  #define SHARED __shared__
  #define INLINE __forceinline__ inline
  #define CONST  __constant__ const

#else

  #define GLOBAL
  #define HOST
  #define DEVICE
  #define SHARED
  #define INLINE inline
  #define CONST  const

#endif

#define COMMON HOST DEVICE

#endif
