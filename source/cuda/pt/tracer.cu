#include <pli_vis/cuda/pt/tracer.h>

#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include <host_defines.h>

#include <pli_vis/cuda/pt/runge_kutta_4_integrator.h>
#include <pli_vis/cuda/pt/trilinear_interpolator.h>
#include <pli_vis/cuda/sh/launch.h>

namespace cupt
{
// Call on a seed_size 1D grid.
template<typename data_type, typename integrator_type>
__global__ void trace_kernel(
  const std::size_t iteration_count,
  const float       step_size      ,
  const uint3       data_dimensions,
  const float3      data_spacing   ,
  const data_type*  data           ,
  const std::size_t seed_size      ,
  const data_type*  seeds          ,
        data_type*  traces         )
{
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= seed_size) return;

  traces[index * iteration_count] = seeds[index];
  integrator_type integrator(data_dimensions, data_spacing, data);
  typename integrator_type::integration_step step(seeds[index], step_size);
  for (auto iteration = 1; iteration < iteration_count; ++iteration)
  {
    integrator.compute_step(step);
    if (step.done())
    {
      traces[index * iteration_count + iteration] = step.end_point;
      step.restart_at(step.end_point);
    }
    else
    {
      step.end_point = step.start_point;
      break;
    }
  }
}

std::vector<std::vector<float3>> trace(
  const std::size_t          iteration_count,
  const float                step_size      ,
  const uint3                data_dimensions,
  const float3               data_spacing   ,
  const std::vector<float3>& data           ,
  const std::vector<float3>& seeds          )
{
  thrust::device_vector<float3> data_gpu   = data ;
  thrust::device_vector<float3> seeds_gpu  = seeds;
  thrust::device_vector<float3> traces_gpu(iteration_count * seeds.size(), float3{0.0f, 0.0f, 0.0f});
  const auto data_gpu_ptr   = raw_pointer_cast(&data_gpu  [0]);
  const auto seeds_gpu_ptr  = raw_pointer_cast(&seeds_gpu [0]);
  const auto traces_gpu_ptr = raw_pointer_cast(&traces_gpu[0]);
  cudaDeviceSynchronize();

  trace_kernel
    <float3, runge_kutta_4_integrator<float3, trilinear_interpolator<float3>>>
    <<<pli::grid_size_1d(seeds.size()), pli::block_size_1d()>>>(
    iteration_count,
    step_size      ,
    data_dimensions,
    data_spacing   ,
    data_gpu_ptr   ,
    seeds.size()   ,
    seeds_gpu_ptr  ,
    traces_gpu_ptr );
  cudaDeviceSynchronize();

  std::vector<float3> traces_linear(traces_gpu.size());
  thrust::copy (traces_gpu.begin(), traces_gpu.end (), traces_linear.begin());
  cudaDeviceSynchronize();

  std::vector<std::vector<float3>> traces(seeds.size());
  for(auto i = 0; i < traces.size(); ++i)
    traces[i] = std::vector<float3>(traces_linear.begin() + i * iteration_count, traces_linear.begin() + (i + 1) * iteration_count);
  return traces;
}
}