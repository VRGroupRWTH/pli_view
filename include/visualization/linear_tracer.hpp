#ifndef PLI_VIS_LINEAR_TRACER_HPP_
#define PLI_VIS_LINEAR_TRACER_HPP_

#include <visualization/tangent/base_operations.hpp>
#include <visualization/tangent/cartesian_grid.hpp>
#include <visualization/tangent/cartesian_locator.hpp>
#include <visualization/tangent/runge_kutta_4_integrator.hpp>
#include <visualization/tangent/simple_tracer.hpp>
#include <visualization/tangent/trace_recorder.hpp>

namespace pli
{
class nearest_neighbor_interpolator final 
{
public:
   nearest_neighbor_interpolator(const tangent::CartesianGrid& data) : data_(data), locator_(data)
  {
     
  }
  ~nearest_neighbor_interpolator() = default;

  bool              IsInside   (const tangent::point_t& point) const 
  {
    return locator_.IsInsideGrid(point);
  }
  tangent::vector_t Interpolate(const tangent::point_t& point) const 
  {
    assert(locator_.IsInsideGrid(point));

    tangent::vector_t previous, next;
    tangent::cell_vectors_t vectors;
    auto max_dot = -1.0F;
    auto cells   = locator_.GetCellPointIds(locator_.GetGridCoords(point));
    for (std::size_t p = 0; p < 8; ++p)
    {
      vectors[p] = data_.GetVectorValue(cells[p]);
      auto dot = std::inner_product(previous.begin(), previous.end(), vectors[p].begin(), 0);
      if (dot > max_dot)
        next = vectors[p];
    }
    return next;
  }

private:
  const tangent::CartesianGrid&   data_   ;
  const tangent::CartesianLocator locator_;
};
struct nearest_neighbor_interpolation_trait 
{
  using Data         = tangent::CartesianGrid;
  using Interpolator = nearest_neighbor_interpolator;
};
struct linear_tracer_trait
{
  using Data         = tangent::CartesianGrid;
  using Integrator   = tangent::RungeKutta4Integrator<nearest_neighbor_interpolation_trait>;
  using Recorder     = tangent::TraceRecorder;
};
using  linear_tracer = tangent::SimpleTracer<linear_tracer_trait>;
}

#endif