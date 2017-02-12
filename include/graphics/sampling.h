#ifndef PLI_VIS_SAMPLING_H_
#define PLI_VIS_SAMPLING_H_

#include <array>
#include <vector>

#include <boost/multi_array.hpp>

namespace pli
{
std::vector<std::array<float, 3>> sample_sphere(
  const std::array<size_t, 2>& sample_dimensions);

boost::multi_array<std::array<float, 3>, 4> sample_sums(
  const boost::multi_array<float, 4>& coefficients,
  const std::array<size_t, 2>&        sample_dimensions);
}

#endif