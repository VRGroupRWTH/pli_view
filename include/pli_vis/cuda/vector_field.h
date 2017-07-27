#ifndef PLI_VIS_VECTOR_FIELD_H_
#define PLI_VIS_VECTOR_FIELD_H_

#include <functional>
#include <string>

#include <vector_types.h>

namespace pli
{
void create_vector_field(
  const uint3&  dimensions,
  const float3* vectors   ,
  const float&  scale     ,
        float3* points    ,
        float3* colors    ,
  std::function<void(const std::string&)> status_callback = [](const std::string&) {});
}

#endif