//-------------------------------------------------------------------------------
// tangent
//
// Copyright(c) 2017 RWTH Aachen University, Germany,
// Virtual Reality & Immersive Visualisation Group.
//-------------------------------------------------------------------------------
//                                 License
//
// This framework is free software : you can redistribute it and / or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// In the future, we may decide to add a commercial license
// at our own discretion without further notice.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.If not, see < http://www.gnu.org/licenses/>.
//-------------------------------------------------------------------------------
#ifndef RUNTIME_MEASUREMENT_H
#define RUNTIME_MEASUREMENT_H

#include "time_referenced_measurement.hpp"
#include <chrono>
#include <cstdint>
#include <string>

namespace tangent {
/**
 * This class defines simple stop watch semantics, i.e. it measures the time
 * between start and stop.
 * BEWARE: This measurement is NOT THREAD SAFE! So make sure you use a
 * dedicated instance for each thread if running in a threaded environment!
 */
template <typename TimeUnit = std::chrono::milliseconds>
class RuntimeMeasurement final : public TimeReferencedMeasurement<TimeUnit> {
public:
  RuntimeMeasurement(const std::string &title)
      : TimeReferencedMeasurement<TimeUnit>{title} {}
  ~RuntimeMeasurement() = default;

  void Start() { start_ = std::chrono::high_resolution_clock::now(); }

  void Stop() { end_ = std::chrono::high_resolution_clock::now(); }

  std::uint64_t GetTimeTaken() const {
    return std::chrono::duration_cast<TimeUnit>(end_ - start_).count();
  }

  std::ostream &WriteData(std::ostream &out_stream) const override {
    out_stream << "runtime : ";
    out_stream << this->GetLabel() << " : ";
    out_stream << this->GetTimeTaken() << " : ";
    out_stream << this->GetTimeUnitString() << std::endl;
    return out_stream;
  }

protected:
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};
};

#endif