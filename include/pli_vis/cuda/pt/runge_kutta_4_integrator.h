#ifndef CUPT_RUNGE_KUTTA_4_INTEGRATOR_H_
#define CUPT_RUNGE_KUTTA_4_INTEGRATOR_H_

#include <host_defines.h>
#include <vector_types.h>

#include "../utility/vector_ops.h"

namespace cupt 
{
template <typename interpolator_trait> 
class runge_kutta_4_integrator 
{
public:
  using interpolator = typename interpolator_trait::interpolator;
  using data         = typename interpolator_trait::data;

  enum class step_state 
  {
    stage_0  = 0,
    stage_1  = 1,
    stage_2  = 2,
    stage_3  = 3,
    finished = 4
  };

  struct integration_step 
  {
    __host__ __device__ integration_step(const float3& start_point, const float step) 
    : start_point(start_point)
    , end_point  (start_point)
    , current_k  ({0.0f, 0.0f, 0.0f})
    , sum_k      ({0.0f, 0.0f, 0.0f})
    , time_step  (step)
    , next_stage (step_state::stage_0)
    {
      
    }

    __host__ __device__ void restart_at(float3& start) 
    {
      start_point = start;
      sum_k       = {0.0f, 0.0f, 0.0f};
      next_stage  = step_state::stage_0;
    }
    __host__ __device__ bool done      () const
    {
      return next_stage == step_state::finished;
    }

    float3     start_point;
    float3     end_point  ;
    float3     current_k  ;
    float3     sum_k      ;
    float      time_step  ;
    step_state next_stage ;
  };

  __host__ __device__ explicit runge_kutta_4_integrator(const data* data) : interpolator_(data), stage_factors_({0.0f, 0.5f, 0.5f}), result_factors_({1.0f, 2.0f, 2.0f}) { }
  __host__ __device__         ~runge_kutta_4_integrator() = default;

  __host__ __device__ void set_data    (const data* data)
  {
    interpolator_ = interpolator(data);
  }
  __host__ __device__ void compute_step(integration_step& step_data) const 
  {
    for (auto stage = step_data.next_stage; stage < step_state::finished; ++stage) 
    {
      float3 current_point = step_data.start_point + stage_factors_[stage] * step_data.current_k;

      if (!interpolator_.is_valid(current_point))
        return;
      
      float3 current_vector = interpolator_.interpolate(current_point);
      step_data.current_k   = step_data.time_step * current_vector;
      step_data.sum_k       = step_data.sum_k + result_factors_[stage] * step_data.current_k;
      step_data.next_stage  = stage + 1;
    }
    step_data.end_point = step_data.start_point + 1.0f / 6.0f * step_data.sum_k;
  }

private:
  interpolator interpolator_  ;
  float3       stage_factors_ ;
  float3       result_factors_;
};
}

#endif
