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
#ifndef AVERAGE_DURATION_MEASUREMENT_H
#define AVERAGE_DURATION_MEASUREMENT_H

#include "counter_measurement.hpp"
#include "time_referenced_measurement.hpp"
#include <chrono>
#include <cstdint>
#include <typeinfo>

namespace tangent {
template <typename TimeUnit = std::chrono::milliseconds>
/**
 * This measurement takes to cumulative time of some event, i.e. the sum of
 * times over multiple episodes.
 * BEWARE: This measurement is NOT THREAD SAFE! So make sure you use a
 * dedicated instance for each thread if running in a threaded environment!
 */
class CummulativeTimeMeasurement final
    : public TimeReferencedMeasurement<TimeUnit> {
public:
  CummulativeTimeMeasurement(const std::string &title)
      : TimeReferencedMeasurement<TimeUnit>{title}, counter_{""},
        total_time_taken_{0} {}
  ~CummulativeTimeMeasurement() = default;

  void StartEpisode() { start_ = std::chrono::high_resolution_clock::now(); }

  void StopEpisode() {
    total_time_taken_ +=
        (std::chrono::high_resolution_clock::now() - start_).count();
    counter_.Count();
  }

  void Reset() { total_time_taken_ = 0; }

  std::uint64_t GetCummulativeTime() const { return total_time_taken_; }
  std::uint64_t GetNumberOfEpisodes() const { return counter_.GetCount(); }
  float GetAverageEpisodeDuration() const {
    return static_cast<float>(total_time_taken_) /
           static_cast<float>(counter_.GetCount());
  }

  std::ostream &WriteData(std::ostream &out_stream) const override {
    out_stream << "cummulative_time : ";
    out_stream << this->GetLabel() << " : ";
    out_stream << total_time_taken_ << " : ";
    out_stream << this->GetTimeUnitString() << std::endl;
    out_stream << "cummulative_count : ";
    out_stream << this->GetLabel() << " : ";
    out_stream << counter_.GetCount() << " : ";
    out_stream << "count" << std::endl;
    return out_stream;
  }

protected:
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::uint64_t total_time_taken_;
  CounterMeasurement counter_;
};
};

#endif