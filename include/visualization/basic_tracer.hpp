#ifndef PLI_VIS_BASIC_TRACER_HPP_
#define PLI_VIS_BASIC_TRACER_HPP_

#include <memory>

#include <boost/multi_array.hpp>

#include <tangent-base/basic_trilinear_interpolator.hpp>
#include <tangent-base/cartesian_grid.hpp>
#include <tangent-base/dummy_recorder.hpp>
#include <tangent-base/runge_kutta_4_integrator.hpp>
#include <tangent-base/simple_tracer.hpp>

#include <attributes/renderable.hpp>
#include <opengl/all.hpp>

namespace pli
{
class basic_tracer : public renderable
{
public:
  struct trilinear_interpolation_trait
  {
    using Data         = tangent::CartesianGrid;
    using Interpolator = tangent::BasicTrilinearInterpolator;
  };
  struct linear_tracer_trait
  {
    using Data       = tangent::CartesianGrid;
    using Integrator = tangent::RungeKutta4Integrator<trilinear_interpolation_trait>;
    using Recorder   = tangent::DummyRecorder;
  };
  using  linear_tracer = tangent::SimpleTracer<linear_tracer_trait>;

  void trace(const boost::multi_array<float, 4>& vectors);

  void initialize()                     override;
  void render    (const camera* camera) override;

private:
  std::unique_ptr<gl::program>      shader_program_;
  std::unique_ptr<gl::vertex_array> vertex_array_  ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_ ;
  std::unique_ptr<gl::array_buffer> color_buffer_  ;
  std::unique_ptr<gl::index_buffer> index_buffer_  ;
  std::size_t                       draw_count_    = 0;
};
}

#endif