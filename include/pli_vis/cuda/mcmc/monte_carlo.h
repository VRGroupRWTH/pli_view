#ifndef PLI_VIS_MONTE_CARLO_H_
#define PLI_VIS_MONTE_CARLO_H_

#include <curand_kernel.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace pli
{
namespace monte_carlo
{
template<typename data_type, typename sample_function_type, typename reduce_function_type = thrust::plus<data_type>>
data_type estimate(
  const unsigned&            thread_count            ,
  const unsigned&            samples_per_thread_count,
  const sample_function_type sample_function         ,
  const reduce_function_type reduce_function         = thrust::plus<data_type>())
{
  return thrust::transform_reduce(
    thrust::counting_iterator<unsigned>(0),
    thrust::counting_iterator<unsigned>(thread_count),
    [samples_per_thread_count, sample_function] __device__ (unsigned thread_id) -> data_type
    {
      curandState state;
      curand_init(thread_id, 0, 0, &state);

      data_type sum;
      for (auto i = 0; i < samples_per_thread_count; i++)
        sum += sample_function(state);
      return sum / samples_per_thread_count;
    },
    data_type(), reduce_function) / thread_count;
}
}
}

#endif
