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
#ifndef TANGENT_BASE_RUNTIME_EXPERIMENT_HPP
#define TANGENT_BASE_RUNTIME_EXPERIMENT_HPP

#include "measurement_base.hpp"
#include "runtime_measurement.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>

namespace tangent {
/**
 * Configure your custom performance experiment by providing callbacks to the
 * three essential phases setup, run, and teardown. In order to account for a
 * wide variety of measurement scenarios, the time units in which the
 * measurement will be taken, can be configured via a template parameter.
 *
 * The runtime of all three phases will be measured per default. Additional
 * measurements can be added as needed. However, it is the client's
 * responsibility to properly inject these measurements into the overall setup,
 * i.e. see to them measurements being called from within the measured code. In
 * order to count the number of integrations, for example, a point recorder can
 * be used (see IntegrationCountingRecorder).
 *
 * Along with executing the three phases outline above, this class provides
 * structured output for the overall experiment.
 */
template <typename TimeUnit = std::chrono::milliseconds>
class PerformanceExperiment {
public:
  using RuntimeMeasurementType = RuntimeMeasurement<TimeUnit>;

  PerformanceExperiment(const std::string &title,
                        const std::function<void()> &setup,
                        const std::function<void()> &run,
                        const std::function<void()> &teardown)
      : title_{title}, setup_callback_{setup}, run_callback_{run},
        teardown_callback_{teardown},
        setup_timer_{std::make_unique<RuntimeMeasurementType>("setup_time")},
        run_timer_{std::make_unique<RuntimeMeasurementType>("run_time")},
        teardown_timer_{
            std::make_unique<RuntimeMeasurementType>("teardown_time")} {
    measurements_.reserve(10);
  }
  ~PerformanceExperiment() = default;

  void AddMeasurement(MeasurementBase *measurement) {
    measurements_.push_back(measurement);
  }

  void Run() {
    this->ExecuteStep(setup_callback_, setup_timer_.get());
    this->ExecuteStep(run_callback_, run_timer_.get());
    this->ExecuteStep(teardown_callback_, teardown_timer_.get());
  }

  std::uint64_t GetSetupTime() const { return setup_timer_->GetTimeTaken(); }

  std::uint64_t GetRuntime() const { return run_timer_->GetTimeTaken(); }

  std::uint64_t GetTeardownTime() const {
    return teardown_timer_->GetTimeTaken();
  }

  template <typename MeasurementType>
  MeasurementType *GetMeasurementByTitle(const std::string &title) const {
    for (auto m : measurements_) {
      if (m->GetLabel() == title)
        return dynamic_cast<MeasurementType *>(m);
    }
    return nullptr;
  }

  std::vector<MeasurementBase *> GetMeasurements() const {
    std::vector<MeasurementBase *> result;
    return measurements_;
  }

  std::ostream &WriteData(std::ostream &out_stream) const {
    this->WriteHeader(out_stream);
    this->WriteDefaultTimes(out_stream);
    this->WriteMeasurements(out_stream);
    return out_stream;
  }

protected:
  void ExecuteStep(const std::function<void()> &step,
                   RuntimeMeasurementType *timer) {
    timer->Start();
    step();
    timer->Stop();
  }

  void WriteHeader(std::ostream &out_stream) const {
    out_stream << "# tangent experiment data" << std::endl;
    out_stream << "# EXPRIMENT: " << title_ << std::endl;
    out_stream << "# Format: " << std::endl;
    out_stream << "# measurement type : measurement label : result : unit"
               << std::endl;
    out_stream << std::endl;
  }

  void WriteDefaultTimes(std::ostream &out_stream) const {
    setup_timer_->WriteData(out_stream);
    run_timer_->WriteData(out_stream);
    teardown_timer_->WriteData(out_stream);
  }

  void WriteMeasurements(std::ostream &out_stream) const {
    for (const auto &m : measurements_)
      m->WriteData(out_stream);
    out_stream << std::endl;
  }

private:
  const std::string title_;

  std::function<void()> setup_callback_;
  std::function<void()> run_callback_;
  std::function<void()> teardown_callback_;

  using RuntimeMeasurePtr = std::unique_ptr<RuntimeMeasurementType>;
  RuntimeMeasurePtr setup_timer_;
  RuntimeMeasurePtr run_timer_;
  RuntimeMeasurePtr teardown_timer_;

  std::vector<MeasurementBase *> measurements_;
};
}

#endif // Include guard.
