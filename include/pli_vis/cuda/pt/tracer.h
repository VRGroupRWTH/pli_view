#ifndef CUPT_TRACER_H_
#define CUPT_TRACER_H_

#include <host_defines.h>
#include <vector_types.h>

namespace cupt 
{
template <typename tracer_trait> class tracer {
public:
  using data             = typename tracer_trait::data;
  using integrator       = typename tracer_trait::integrator;
  using integration_step = typename integrator::integration_step;

  __host__ __device__ explicit tracer(const data* data, float step_size = 0.01f, std::size_t iteration_count = 100) : data_(data), step_size_(step_size), iteration_count_(iteration_count) { }

  // TODO: Convert to a kernel.
  __host__ __device__ void trace(
    const std::size_t seed_count, 
    const float3*     seeds     , 
    float3**          traces    ) // Preallocated seed_count x iteration_count 2D array.
  {
    integrator integrator(data_);
    for (auto seed_id = 0; seed_id < seed_count; ++seed_id)
    {
      traces[seed_id][0] = seeds[seed_id];

      integration_step step(seeds[seed_id], step_size_);
      for (auto iteration = 1; iteration < iteration_count_; ++iteration)
      {
        integrator.compute_step(step);
        if (step.done())
        {
          traces[seed_id][iteration] = step.end_point;
          step.restart_at(step.end_point);
        }
        else
        {
          step.end_point = step.start_point;
          break;
        }
      }
    }
  }

private:
  const data* data_           ;
  float       step_size_      ;
  std::size_t iteration_count_;
};
}

#endif
