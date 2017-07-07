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
#ifndef COUNTER_MEASUREMENT_H
#define COUNTER_MEASUREMENT_H

#include "measurement_base.hpp"
#include <cstdint>
#include <string>

namespace tangent {
/**
 * This is a simple measurement class to count occurrences of events and
 * integrate them into a measurement output.
 * BEWARE: The count method is NOT THREAD SAFE! So make sure you use a
 * dedicated counter for each thread if running in a threaded environment!
 */
class CounterMeasurement final : public MeasurementBase {
public:
  CounterMeasurement(const std::string &title)
      : MeasurementBase{title}, counter_{0} {}
  ~CounterMeasurement() = default;

  void Count() { ++counter_; }

  std::uint64_t GetCount() const { return counter_; }
  void SetCount(const std::uint64_t count) { counter_ = count; }

  std::ostream &WriteData(std::ostream &out_stream) const override {
    out_stream << "counter : ";
    out_stream << this->GetLabel() << " : ";
    out_stream << counter_ << " : ";
    out_stream << "count" << std::endl;
    return out_stream;
  }

private:
  std::uint64_t counter_;
};
};

#endif