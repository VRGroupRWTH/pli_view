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
#ifndef TIME_REFERENCED_MEASUREMENT_H
#define TIME_REFERENCED_MEASUREMENT_H

#include "time_referenced_measurement.hpp"

#include <chrono>
#include <string>
#include <typeinfo>

namespace tangent {
/**
 * Base class for all measurements using time units. Defines a common output
 * format for unit strings.
 */
template <typename TimeUnit>
class TimeReferencedMeasurement : public MeasurementBase {
public:
  TimeReferencedMeasurement(const std::string &label)
      : MeasurementBase{label} {}
  virtual ~TimeReferencedMeasurement() = default;

protected:
  std::string GetTimeUnitString() const {
    if (typeid(TimeUnit) == typeid(std::chrono::nanoseconds))
      return "ns";
    if (typeid(TimeUnit) == typeid(std::chrono::microseconds))
      return "mus";
    if (typeid(TimeUnit) == typeid(std::chrono::milliseconds))
      return "ms";
    if (typeid(TimeUnit) == typeid(std::chrono::seconds))
      return "s";
    if (typeid(TimeUnit) == typeid(std::chrono::minutes))
      return "m";
    if (typeid(TimeUnit) == typeid(std::chrono::hours))
      return "ns";
    if (typeid(TimeUnit) == typeid(std::chrono::nanoseconds))
      return "ns";
    return "undefined";
  }

private:
};
};

#endif