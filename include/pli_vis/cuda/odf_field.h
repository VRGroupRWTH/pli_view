#ifndef PLI_VIS_ODF_FIELD_H_
#define PLI_VIS_ODF_FIELD_H_

#include <functional>
#include <math.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include <pli_vis/cuda/sh/vector_ops.h>

namespace pli
{
void calculate_odfs(
  cublasHandle_t     cublas        ,
  cusolverDnHandle_t cusolver      ,
  const uint3&       dimensions    ,
  const uint3&       vectors_size  , 
  const uint2&       histogram_bins,
  const unsigned     maximum_degree,
  const float*       directions    ,
  const float*       inclinations  ,
        float*       coefficients  ,
        std::function<void(const std::string&)> status_callback = [](const std::string&){});

void calculate_odfs(
  cublasHandle_t     cublas        ,
  cusolverDnHandle_t cusolver      ,
  const uint3&       dimensions    ,
  const uint3&       vectors_size  , 
  const uint2&       histogram_bins,
  const unsigned     maximum_degree,
  const float3*      unit_vectors  ,
        float*       coefficients  ,
        std::function<void(const std::string&)> status_callback = [](const std::string&){});

void sample_odfs(
  const uint3&       dimensions        ,
  const unsigned     coefficient_count ,
  const float*       coefficients      ,
  const uint2&       tessellations     , 
  const float3&      vector_spacing    ,
  const uint3&       vector_dimensions ,
  const float        scale             ,
        float3*      points            ,
        float4*      colors            ,
        unsigned*    indices           ,
        bool         clustering        = false,
        float        cluster_threshold = 0.0  ,
        std::function<void(const std::string&)> status_callback = [](const std::string&){});
  
// Called on a layer_dimensions.x x layer_dimensions.y x layer_dimensions.z 3D grid.
__global__ void sample_odf_layer(
  const uint3    layer_dimensions  ,
  const unsigned layer_offset      ,
  const unsigned coefficient_count ,
  float*         coefficients      ,
  bool           is_2d             ,
  bool           clustering        ,
  float          cluster_threshold );

template<typename vector_type, typename scalar_type>
__global__ void quantify_and_project(
  // Input related parameters.
        uint3          dimensions         ,
        uint3          vectors_size       ,
  // Quantification related parameters. 
  const vector_type*   vectors            , 
        uint2          histogram_bins     ,
        vector_type*   histogram_vectors  ,
  // Projection related parameters.     
        unsigned       maximum_degree     ,
  const scalar_type*   inverse_transform  ,
        scalar_type*   coefficient_vectors);
}

#endif