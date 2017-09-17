#ifndef PLI_VIS_ODF_FIELD_H_
#define PLI_VIS_ODF_FIELD_H_

#include <functional>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector_types.h>

namespace pli
{
void calculate_odfs(
  cublasHandle_t     cublas            ,
  cusolverDnHandle_t cusolver          ,
  const uint3&       dimensions        ,
  const uint3&       vectors_dimensions,
  const uint2&       histogram_bins    ,
  const unsigned     maximum_degree    ,
  const float3*      vectors           ,
        float*       coefficients      ,
        bool         even_only         ,
        std::function<void(const std::string&)> status_callback = [](const std::string&){});

void sample_odfs(
  const uint3&       dimensions        ,
  const unsigned     maximum_degree    ,
  const float*       coefficients      ,
  const uint2&       tessellations     , 
  const uint3&       vector_dimensions ,
  const float        scale             ,
        float3*      points            ,
        float3*      directions        ,
        unsigned*    indices           ,
        bool         hierarchical      = false,
        bool         clustering        = false,
        float        cluster_threshold = 0.0  ,
        std::function<void(const std::string&)> status_callback = [](const std::string&){});

void extract_peaks(
  const uint3&       dimensions    ,
  const unsigned     maximum_degree,
  const float*       coefficients  ,
  const uint2&       tessellations , 
  const unsigned     maxima_count  ,
        float3*      maxima        ,
        std::function<void(const std::string&)> status_callback = [](const std::string&){});
}

#endif