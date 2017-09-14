#ifndef CUPT_CARTESIAN_STREAMLINE_TRACER_H_
#define CUPT_CARTESIAN_STREAMLINE_TRACER_H_

#include <vector_types.h>

#include "cartesian_grid.h"
#include "runge_kutta_4_integrator.h"
#include "tracer.h"
#include "trilinear_interpolator.h"

namespace cupt 
{
template<class type = float3>
struct trilinear_interpolation_trait 
{
  using data         = cartesian_grid        <type>;
  using interpolator = trilinear_interpolator<type>;
};

template<class type = float3>
struct cartesian_streamline_trait 
{
  using data       = cartesian_grid<type>;
  using integrator = runge_kutta_4_integrator<trilinear_interpolation_trait<type>>;
};

using cartesian_streamline_tracer = tracer<cartesian_streamline_trait<float3>>;
}

#endif