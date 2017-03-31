#ifndef PLI_VIS_LINEAR_TRACER_HPP_
#define PLI_VIS_LINEAR_TRACER_HPP_

#include <visualization/tangent/basic_trilinear_interpolator.hpp>
#include <visualization/tangent/cartesian_grid.hpp>
#include <visualization/tangent/runge_kutta_4_integrator.hpp>
#include <visualization/tangent/simple_tracer.hpp>
#include <visualization/tangent/trace_recorder.hpp>

namespace pli
{
struct trilinear_interpolation_trait 
{
  using Data         = tangent::CartesianGrid;
  using Interpolator = tangent::BasicTrilinearInterpolator;
};
struct linear_tracer_trait
{
  using Data         = tangent::CartesianGrid;
  using Integrator   = tangent::RungeKutta4Integrator<trilinear_interpolation_trait>;
  using Recorder     = tangent::TraceRecorder;
};
using  linear_tracer = tangent::SimpleTracer<linear_tracer_trait>;
}

#endif