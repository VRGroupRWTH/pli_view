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
#ifndef TANGENT_BASE_TRACE_RECORDER_HPP
#define TANGENT_BASE_TRACE_RECORDER_HPP

#include "base_types.hpp"
#include "particle_population.hpp"
#include <cassert>

namespace tangent {
/**
 * A TraceRecorder is used to record trace points in a straightforward vector of
 * vectors layout. The underlying data type is a tangent::ParticlePopulation.
 * Each point that is issued to the recorder will just be appended to the
 * respective particle trace's representation.
 */
class TraceRecorder final {
public:
  TraceRecorder() = default;
  ~TraceRecorder() = default;

  void InitRecording(const std::size_t num_traces,
                     const std::size_t expected_trace_length) {
    particles_ = ParticlePopulation{num_traces, expected_trace_length};
  }

  void InitTrace(const std::size_t trace_id, const point_t &seed) {
    assert(trace_id < particles_.GetNumberOfTraces());
    auto &trace = particles_[trace_id];
    trace.clear();
    trace.push_back(seed);
  }

  void AppendTracePoint(const std::size_t trace_id, const point_t &point) {
    assert(trace_id < particles_.GetNumberOfTraces());
    particles_[trace_id].push_back(point);
  }

  ParticlePopulation &GetPopulation() { return particles_; }

private:
  ParticlePopulation particles_;
};
}

#endif // Include guard.
