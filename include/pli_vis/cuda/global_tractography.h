#ifndef PLI_VIS_GLOBAL_TRACTOGRAPHY_H_
#define PLI_VIS_GLOBAL_TRACTOGRAPHY_H_

#include <vector_types.h>

#include <pli_vis/cuda/sh/vector_ops.h>

namespace pli
{
template<
  typename scalar_precision = float , 
  typename vector_precision = float3>
scalar_precision interaction_potential(
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
scalar_precision internal_energy(
  const unsigned          count         , 
  const vector_precision* lhs_positions , 
  const vector_precision* lhs_directions,
  const vector_precision* rhs_positions ,
  const vector_precision* rhs_directions,
  const scalar_precision& bias          )
{
  scalar_precision value(0);
  for(auto i = 0; i < count; i++)
    value += interaction_potential(
      lhs_positions [i], 
      lhs_directions[i], 
      rhs_positions [i], 
      rhs_directions[i], 
      bias);
  return value;
}
}

#endif
