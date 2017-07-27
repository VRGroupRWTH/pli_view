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
#ifndef FREQUENCY_MEASUREMENT_H
#define FREQUENCY_MEASUREMENT_H

#include "counter_measurement.hpp"
#include "runtime_measurement.hpp"
#include "time_referenced_measurement.hpp"

#include <string>

namespace tangent {
/**
 * This class allows clients to track frequencies, i.e. number of events per
 * unit time.
 * BEWARE: This measurement is NOT THREAD SAFE! So make sure you use a
 * dedicated instance for each thread if running in a threaded environment!
 */
template <typename TimeUnit = std::chrono::milliseconds>
class FrequencyMeasurement final : public TimeReferencedMeasurement<TimeUnit> {
public:
  FrequencyMeasurement(const std::string title,
                       const RuntimeMeasurement<TimeUnit> &time,
                       const CounterMeasurement &count)
      : TimeReferencedMeasurement<TimeUnit>{title}, time_{time}, count_{count} {
  }
  ~FrequencyMeasurement() = default;

  float GetFrequency() const {
    return static_cast<float>(count_.GetCount()) / time_.GetTimeTaken();
  }

  std::ostream &WriteData(std::ostream &out_stream) const override {
    out_stream << "frequency_time : ";
    out_stream << this->GetLabel() << " : ";
    out_stream << time_.GetTimeTaken() << " : ";
    out_stream << this->GetTimeUnitString() << std::endl;
    out_stream << "frequency_count : ";
    out_stream << this->GetLabel() << " : ";
    out_stream << count_.GetCount() << " : ";
    out_stream << "count" << std::endl;
    return out_stream;
  }

private:
  const RuntimeMeasurement<TimeUnit> &time_;
  const CounterMeasurement &count_;
};
};

#endif