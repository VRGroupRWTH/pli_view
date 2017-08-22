#ifndef PLI_VIS_GLOBAL_TRACTOGRAPHY_H_
#define PLI_VIS_GLOBAL_TRACTOGRAPHY_H_

#include <vector_types.h>

#include <pli_vis/cuda/sh/spherical_harmonics.h>
#include <pli_vis/cuda/sh/vector_ops.h>

namespace pli
{
template<
  typename scalar_precision = float , 
  typename vector_precision = float3>
  __host__ __device__ scalar_precision interaction_potential(
  const vector_precision& lhs_position  ,
  const vector_precision& lhs_direction ,
  const vector_precision& rhs_position  ,
  const vector_precision& rhs_direction ,
  const scalar_precision& bias          ,
  const scalar_precision& length        = 1)
{
  auto midpoint    = (lhs_position + rhs_position) / 2;
  auto lhs_sign    = dot(lhs_direction, midpoint - lhs_position) >= 0 ? 1 : 0;
  auto rhs_sign    = dot(rhs_direction, midpoint - rhs_position) >= 0 ? 1 : 0;
  auto lhs_segment = lhs_position + lhs_sign * length * lhs_direction;
  auto rhs_segment = rhs_position + rhs_sign * length * rhs_direction;
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
  const unsigned          count         , 
  const vector_precision* lhs_positions , 
  const vector_precision* lhs_directions,
  const vector_precision* rhs_positions ,
  const vector_precision* rhs_directions,
  const scalar_precision& bias          ,
  const scalar_precision& length        = 1)
{
  scalar_precision value(0);
  for(auto i = 0; i < count; i++)
    value += interaction_potential(
      lhs_positions [i], 
      lhs_directions[i], 
      rhs_positions [i], 
      rhs_directions[i], 
      bias,
      length);
  return value;
}

// Given 2 adjacent "original" ODFs represented as the coefficients of a spherical harmonic expansion:
// - Monte Carlo sample the ODFs to extract 2 groups of vectors.
// - Calculate the interaction potential of each combination of the vectors and sum them to obtain the internal energy.
// - Project the sampled line segments to two "predicted" ODFs using ASGB2016, HDAG2017.
// - Calculate the L2-distance between the "original" and "predicted" ODFs in order to obtain the external energy.
// - Maximize the sum of internal and external energies using Markov chain.

template<
  typename scalar_precision = float ,
  typename vector_precision = float3>
scalar_precision calculate_predicted_signal(
  const vector_precision& position  ,
  const vector_precision& direction ,
  const scalar_precision& c         ,
  const scalar_precision& sigma     )
{
  
}

template<
  typename scalar_precision = float >
scalar_precision external_energy(
  const unsigned          odf_count     ,
  const unsigned          odf_max_degree,
  const scalar_precision* original_odfs ,
  const scalar_precision* predicted_odfs,
  const scalar_precision  weight        )
{
  auto odf_coefficient_count = coefficient_count(odf_max_degree);
  for(auto i = 0; i < odf_count; ++i)
  {
    auto start_index = odf_coefficient_count * i;
    for(auto j = 0; j < odf_coefficient_count; ++j)
    {
      l2_distance(odf_coefficient_count, original_odfs[start_index], predicted_odfs[start_index]);
    }
  }
}
}

#endif
