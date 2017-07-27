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
#ifndef TANGENT_BASE_DUMMY_RECORDER_HPP
#define TANGENT_BASE_DUMMY_RECORDER_HPP

#include "base_types.hpp"

namespace tangent {
/**
 * This recorder can be used as a blank stand in for a real recorder in case
 * that recording particle positions is not required for the task at hand. Not
 * recording anything "on the go" might significantly increase tracing
 * performance, because there will essentially be no write-backs to main memory.
 */
class DummyRecorder final {
public:
  DummyRecorder() = default;
  ~DummyRecorder() = default;

  void InitRecording(const std::size_t num_traces,
                     const std::size_t expected_trace_length){};

  void InitTrace(const std::size_t trace_id, const point_t &seed) { return; }

  void AppendTracePoint(const std::size_t trace_id, const point_t &point) {
    return;
  }
};
}

#endif // Include guard.
