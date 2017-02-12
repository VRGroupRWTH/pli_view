#include /* implements */ <graphics/sampling.hpp>

#define _USE_MATH_DEFINES

#include <math.h>

#include <convert.hpp>
#include <cush.h>

namespace pli
{
std::vector<std::array<float, 3>>           sample_sphere
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
  (const boost::multi_array<float, 4>& coefficients,
  const std::array<size_t, 2>&         dimensions  )
{
  auto shape = coefficients.shape();

  boost::multi_array<std::array<float, 3>, 4> samples(boost::extents
    [shape[0]]
    [shape[1]]
    [shape[2]]
    [dimensions[0] * dimensions[1]]);

  for (auto x = 0; x < shape[0]; x++)
  {
    for (auto y = 0; y < shape[1]; y++)
    {
      for (auto z = 0; z < shape[2]; z++)
      {
        for (auto lon = 0; lon < dimensions[0]; lon++)
        {
          for (auto lat = 0; lat < dimensions[1]; lat++)
          {
            auto& sample = samples[x][y][z][lon * dimensions[1] + lat];
            sample[1] = 2 * M_PI * lon /  dimensions[0];
            sample[2] =     M_PI * lat / (dimensions[1] - 1);
            for (auto c = 0; c < shape[3]; c++)
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
