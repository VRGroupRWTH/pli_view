#ifndef PLI_VIS_CONVERT_HPP_
#define PLI_VIS_CONVERT_HPP_

#include <math.h>

#include <vector_types.h>

namespace pli
{
template<typename input_type, typename output_type = input_type>
__host__ __device__ output_type to_spherical_coords(const input_type& input)
{
  output_type output;
  output[0] = sqrt (pow(input[0], 2) + pow(input[1], 2) + pow(input[2], 2));
  output[1] = atan2(input[1] , input [0]);
  output[2] = acos (input[2] / output[0]);
  return output;
}
template<typename input_type, typename output_type = input_type>
__host__ __device__ output_type to_cartesian_coords(const input_type& input)
{
  output_type output;
  output[0] = input[0] * cos(input[1]) * sin(input[2]);
  output[1] = input[0] * sin(input[1]) * sin(input[2]);
  output[2] = input[0] * cos(input[2]);
  return output;
}

#define SPECIALIZE_CONVERT(TYPE)                                         \
template <typename output_type = TYPE>                                   \
__host__ __device__ output_type to_spherical_coords(const TYPE& input)   \
{                                                                        \
  output_type output;                                                    \
  output.x = sqrt (pow(input.x, 2) + pow(input.y, 2) + pow(input.z, 2)); \
  output.y = atan2(input.y, input.x);                                    \
  output.z = acos (input.z / output.x);                                  \
  return output;                                                         \
}                                                                        \
template<typename output_type = TYPE>                                    \
__host__ __device__ output_type to_cartesian_coords(const TYPE& input)   \
{                                                                        \
  output_type output;                                                    \
  output.x = input.x * cos(input.y) * sin(input.z);                      \
  output.y = input.x * sin(input.y) * sin(input.z);                      \
  output.z = input.x * cos(input.z);                                     \
  return output;                                                         \
}                                                                        \
                                        
SPECIALIZE_CONVERT(char3     )
SPECIALIZE_CONVERT(uchar3    )
SPECIALIZE_CONVERT(short3    )
SPECIALIZE_CONVERT(ushort3   )
SPECIALIZE_CONVERT(int3      )
SPECIALIZE_CONVERT(uint3     )
SPECIALIZE_CONVERT(long3     )
SPECIALIZE_CONVERT(ulong3    )
SPECIALIZE_CONVERT(float3    )
SPECIALIZE_CONVERT(longlong3 )
SPECIALIZE_CONVERT(ulonglong3)
SPECIALIZE_CONVERT(double3   )
              
SPECIALIZE_CONVERT(char4     )
SPECIALIZE_CONVERT(uchar4    )
SPECIALIZE_CONVERT(short4    )
SPECIALIZE_CONVERT(ushort4   )
SPECIALIZE_CONVERT(int4      )
SPECIALIZE_CONVERT(uint4     )
SPECIALIZE_CONVERT(long4     )
SPECIALIZE_CONVERT(ulong4    )
SPECIALIZE_CONVERT(float4    )
SPECIALIZE_CONVERT(longlong4 )
SPECIALIZE_CONVERT(ulonglong4)
SPECIALIZE_CONVERT(double4   )

#undef SPECIALIZE_CONVERT
}

#endif
