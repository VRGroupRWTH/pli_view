#ifndef PLI_VIS_GLOBAL_TRACTOGRAPHY_H_
#define PLI_VIS_GLOBAL_TRACTOGRAPHY_H_

#include <vector_types.h>

#include <pli_vis/cuda/sh/spherical_harmonics.h>
#include <pli_vis/cuda/sh/vector_ops.h>

namespace pli
{
// Based on the following:
// - Global reconstruction of neuronal fibres      (Reisert et al. 2009)
// - Global fiber reconstruction becomes practical (Reisert et al. 2011)
template<
  typename scalar_precision = float , 
  typename vector_precision = float3>
__host__ __device__ scalar_precision interaction_potential(
  const vector_precision& lhs_position    ,
  const vector_precision& lhs_orientation ,
  const vector_precision& rhs_position    ,
  const vector_precision& rhs_orientation ,
  const scalar_precision& bias            ,
  const scalar_precision& length          = 1)
{
  auto midpoint    = (lhs_position + rhs_position) / 2;
  auto lhs_sign    = dot(lhs_orientation, midpoint - lhs_position) >= 0 ? 1 : -1;
  auto rhs_sign    = dot(rhs_orientation, midpoint - rhs_position) >= 0 ? 1 : -1;
  auto lhs_segment = lhs_position + lhs_sign * length * lhs_orientation;
  auto rhs_segment = rhs_position + rhs_sign * length * rhs_orientation;
  auto lhs_diff    = lhs_segment - midpoint;
  auto rhs_diff    = rhs_segment - midpoint;
  auto lhs_sq_dist = pow(lhs_diff.x, 2) + pow(lhs_diff.y, 2) + pow(lhs_diff.z, 2);
  auto rhs_sq_dist = pow(rhs_diff.x, 2) + pow(rhs_diff.y, 2) + pow(rhs_diff.z, 2);
  return (1.0 / pow(length, 2)) * (lhs_sq_dist  + rhs_sq_dist) - bias;
}

template<
  typename scalar_precision = float , 
  typename vector_precision = float3>
__host__ __device__ scalar_precision internal_energy(
  const unsigned          count           , 
  const vector_precision* lhs_positions   , 
  const vector_precision* lhs_orientations,
  const vector_precision* rhs_positions   ,
  const vector_precision* rhs_orientations,
  const scalar_precision& bias            ,
  const scalar_precision& length          = 1)
{
  scalar_precision value(0);
  for(auto i = 0; i < count; ++i)
    value += interaction_potential(
      lhs_positions   [i], 
      lhs_orientations[i], 
      rhs_positions   [i], 
      rhs_orientations[i], 
      bias,
      length);
  return value;
}

template<
  typename scalar_precision = float ,
  typename vector_precision = float3>
__host__ __device__ void energy_minimizing_configuration(
  const vector_precision& position        ,
  const vector_precision& direction       ,
        vector_precision& out_position    ,
        vector_precision& out_direction   ,
  const scalar_precision& length          = 1)
{
  out_position  = position + 2 * length * direction;
  out_direction = direction;
}

template<
  typename scalar_precision = float ,
  typename vector_precision = float3>
__host__ __device__ void energy_minimizing_configuration(
  const vector_precision& lhs_position    ,
  const vector_precision& lhs_direction   ,
  const vector_precision& rhs_position    ,
  const vector_precision& rhs_direction   ,
        vector_precision& out_position    ,
        vector_precision& out_direction   ,
  const scalar_precision& length          = 1)
{
  out_position  = (lhs_position + length * lhs_direction + rhs_position + length * rhs_direction) / 2;
  out_direction = normalize(rhs_position - lhs_position);
}

template<
  typename scalar_precision = float ,
  typename vector_precision = float3>
__host__ __device__ scalar_precision predicted_signal(
  const vector_precision& position        ,
  const vector_precision& direction       ,
  const unsigned          count           ,
  const vector_precision* positions       ,
  const vector_precision* directions      ,
  const scalar_precision& weight          ,
  const scalar_precision& c               ,
  const scalar_precision& sigma           )
{
  scalar_precision sum(0);
  for(auto i = 0; i < count; ++i)
  {
    scalar_precision dot_v  = dot(direction, directions[i]);
    vector_precision diff_x = position - positions[i];
    sum += exp(-c * pow(dot_v, 2)) * exp(-dot(diff_x, diff_x) / pow(sigma, 2));
  }
  return weight * sum;
}

template<
  typename scalar_precision = float ,
  typename vector_precision = float3>  
__host__ __device__ void predicted_signal(
  const vector_precision& position        ,
  const unsigned          bin_count       ,
  const vector_precision* bin_directions  ,
  const unsigned          count           ,
  const vector_precision* positions       ,
  const vector_precision* directions      ,
  const scalar_precision& weight          ,
  const scalar_precision& c               ,
  const scalar_precision& sigma           ,
  const scalar_precision& max_degree      ,
        scalar_precision* predicted_odfs  )
{
  const vector_precision bin_magnitudes[count];
  for(auto i = 0; i < bin_count; ++i)
  {
    bin_magnitudes[i] = predicted_signal(
      position         , 
      bin_directions[i], 
      count            , 
      positions        , 
      directions       , 
      weight           , 
      c                , 
      sigma            );
  }
  // TODO:
  // - Interpret the sampled directions and the evaluated signal at that direction as a histogram.
  // - Normalize and project to spherical harmonics of given maximum degree.
}

template<
  typename scalar_precision = float>
__host__ __device__ void external_energy(
  const unsigned          odf_count       ,
  const unsigned          odf_max_degree  ,
  const scalar_precision* original_odfs   ,
  const scalar_precision* predicted_odfs  ,
        scalar_precision* out_energies    )
{
  auto odf_coefficient_count = coefficient_count(odf_max_degree);
  for(auto i = 0; i < odf_count; ++i)
  {
    auto start_index = odf_coefficient_count * i;
    out_energies[i]  = l2_distance(odf_coefficient_count, original_odfs[start_index], predicted_odfs[start_index]);
  }
}

template<
  typename scalar_precision = float>
__host__ __device__ scalar_precision posterior_probability(
  const scalar_precision& internal_energy,
  const scalar_precision& external_energy,
  const scalar_precision& temperature    )
{
  return exp((- internal_energy - external_energy) / temperature);
}
}

#endif
