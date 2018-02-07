//-------------------------------------------------------------------------------
// tangent
//
// Copyright (c) 2017 RWTH Aachen University, Germany,
// Virtual Reality & Immersive Visualisation Group.
//-------------------------------------------------------------------------------
//                                 License
//
// This framework is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// In the future, we may decide to add a commercial license
// at our own discretion without further notice.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//-------------------------------------------------------------------------------
#ifndef TANGENT_BASE_PARTICLE_POPULATION_HPP
#define TANGENT_BASE_PARTICLE_POPULATION_HPP

#include "base_types.hpp"
#include <vector>

namespace tangent {
/**
 * Straightforward data structure to keep track of particle traces. Each trace
 * is represented as a std::vector of points.
 */
class ParticlePopulation final {
public:
  using Trace = std::vector<point_t>;

  ParticlePopulation() = default;
  ParticlePopulation(const std::size_t num_particles,
                     const std::size_t expected_size = 100) {
    traces_.resize(num_particles);
    for (auto &trace : traces_)
      trace.reserve(expected_size);
  }
  ~ParticlePopulation() = default;

  std::size_t GetNumberOfTraces() const { return traces_.size(); }

  Trace &operator[](const std::size_t trace_id) { return traces_[trace_id]; }

protected:
private:
  std::vector<Trace> traces_;
};
}

#endif // Include guard.
