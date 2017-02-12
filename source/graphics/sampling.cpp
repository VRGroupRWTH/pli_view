#include /* implements */ <graphics/sampling.h>

#define _USE_MATH_DEFINES

#include <math.h>

#include <convert.hpp>
#include <cush.h>

namespace pli
{
std::vector<std::array<float, 3>> sample_sphere(const std::array<size_t, 2>& sample_dimensions)
{
  //thrust::device_vector<float3> samples(sample_dimensions[0] * sample_dimensions[1]);
  //cush::sample(0, 0, , raw_pointer_cast(&samples[0]));

  std::vector<std::array<float, 3>> samples_cpu(5);
  //thrust::copy(samples.begin(), samples.end(), samples_cpu.begin());
  return samples_cpu;
}

boost::multi_array<std::array<float, 3>, 4> sample_sums(
  const boost::multi_array<float, 4>& coefficients,
  const std::array<size_t, 2>&        sample_dimensions)
{
  auto shape = coefficients.shape();

  boost::multi_array<std::array<float, 3>, 4> samples(boost::extents
    [shape[0]]
    [shape[1]]
    [shape[2]]
    [sample_dimensions[0] * sample_dimensions[1]]);

  for (auto x = 0; x < shape[0]; x++)
  {
    for (auto y = 0; y < shape[1]; y++)
    {
      for (auto z = 0; z < shape[2]; z++)
      {
        for (auto lon = 0; lon < sample_dimensions[0]; lon++)
        {
          for (auto lat = 0; lat < sample_dimensions[1]; lat++)
          {
            auto& sample = samples[x][y][z][lon * sample_dimensions[1] + lat];
            sample[1] = 2 * M_PI * lon /  sample_dimensions[0];
            sample[2] =     M_PI * lat / (sample_dimensions[1] - 1);

            for (auto c = 0; c < coefficients.size(); c++)
              sample[0] += cush::evaluate(c, sample[1], sample[2]) * coefficients[x][y][z][c];

            sample = pli::to_cartesian_coords(sample);
          }
        }
      }
    }
  }

  return samples;
}
}
