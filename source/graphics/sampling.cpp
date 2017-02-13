#include /* implements */ <graphics/sampling.hpp>

#define _USE_MATH_DEFINES

#include <math.h>

#include <convert.hpp>
#include <cush.h>

#include <cuda/sampler.h>

namespace pli
{
std::vector<std::array<float, 3>> sample_sphere
  (const std::array<size_t, 2>& dimensions)
{
  std::vector<std::array<float, 3>> samples(dimensions[0] * dimensions[1]);

  for (auto lon = 0; lon < dimensions[0]; lon++)
  {
    for (auto lat = 0; lat < dimensions[1]; lat++)
    {
      auto& sample = samples[lon * dimensions[1] + lat];
      sample[0] = 1.0;
      sample[1] = 2 * M_PI * lon /  dimensions[0];
      sample[2] =     M_PI * lat / (dimensions[1] - 1);
      sample    = pli::to_cartesian_coords(sample);
    }
  }

  return samples;
}

boost::multi_array<std::array<float, 3>, 4> sample_sums
  (const boost::multi_array<float, 4>& coefficients     ,
  const std::array<size_t, 2>&         sample_dimensions)
{
  auto shape = coefficients.shape();

  boost::multi_array<std::array<float, 3>, 4> samples(boost::extents
    [shape[0]]
    [shape[1]]
    [shape[2]]
    [sample_dimensions[0] * sample_dimensions[1]]);
  
  boost::multi_array<unsigned, 4> indices(boost::extents
    [shape[0]]
    [shape[1]]
    [shape[2]]
    [4 * sample_dimensions[0] * sample_dimensions[1]]);

  uint3 dimensions        = { shape[0], shape[1], shape[2] };
  uint2 output_dimensions = { sample_dimensions[0], sample_dimensions[1] };
  sample(dimensions, cush::maximum_degree(shape[3]), output_dimensions, coefficients.data(), (float3*) samples.data(), indices.data());

  return samples;
}
}
