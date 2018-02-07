#ifndef CUPT_RUNGE_KUTTA_4_INTEGRATOR_H_
#define CUPT_RUNGE_KUTTA_4_INTEGRATOR_H_

#include <host_defines.h>
#include <vector_types.h>

#include "../utility/vector_ops.h"

namespace cupt 
{
template <class data_type, class interpolator_type> 
class runge_kutta_4_integrator 
{
public:
  struct integration_step 
  {
    __host__ __device__ integration_step(const data_type& start_point, const float step)
    : start_point(start_point)
    , end_point  (start_point)
    , current_k  ({0.0f, 0.0f, 0.0f})
    , sum_k      ({0.0f, 0.0f, 0.0f})
    , time_step  (step)
    , next_stage (0)
    {
      
    }

    __host__ __device__ void restart_at(data_type& start)
    {
      start_point = start;
      sum_k       = {0.0f, 0.0f, 0.0f};
      next_stage  = 0;
    }
    __host__ __device__ bool done      () const
    {
      return next_stage == 4;
    }

    data_type  start_point;
    data_type  end_point  ;
    data_type  current_k  ;
    data_type  sum_k      ;
    float      time_step  ;
    char       next_stage ;
  };

  __host__ __device__ explicit runge_kutta_4_integrator(const uint3& dimensions, const float3& spacing, const data_type* data)
  : interpolator_  (dimensions, spacing, data)
  , stage_factors_ ({0.0f, 0.5f, 0.5f, 1.0f} )
  , result_factors_({1.0f, 2.0f, 2.0f, 1.0f} )
  {
    
  }

  __host__ __device__ void set_data    (const uint3& dimensions, const float3& spacing, const data_type*  data)
  {
    interpolator_ = interpolator_type(dimensions, spacing, data);
  }
  __host__ __device__ void compute_step(integration_step& step_data) const 
  {
    for (auto stage = 0; stage < 4; ++stage) 
    {
      auto stage_factor = 0.0f, result_factor = 0.0f;
      if      (stage == 0) { stage_factor = stage_factors_.x; result_factor = result_factors_.x; }
      else if (stage == 1) { stage_factor = stage_factors_.y; result_factor = result_factors_.y; }
      else if (stage == 2) { stage_factor = stage_factors_.z; result_factor = result_factors_.z; }
      else if (stage == 3) { stage_factor = stage_factors_.w; result_factor = result_factors_.w; }

      float3 current_point = step_data.start_point + stage_factor * step_data.current_k;

      if (!interpolator_.is_valid(current_point))
        return;
      
      float3 current_vector = interpolator_.interpolate(current_point);
      step_data.current_k   = step_data.time_step * current_vector;
      step_data.sum_k       = step_data.sum_k + result_factor * step_data.current_k;
      step_data.next_stage  = stage + 1;
    }
    step_data.end_point = step_data.start_point + 1.0f / 6.0f * step_data.sum_k;
  }

private:
  interpolator_type interpolator_  ;
  float4            stage_factors_ ;
  float4            result_factors_;
};
}

#endif