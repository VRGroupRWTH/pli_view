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
#ifndef TRACER_BASE_H
#define TRACER_BASE_H

#include "base_types.hpp"
#include "base_operations.hpp"

#include <cassert>
#include <vector>

namespace tangent {

/**
 * TracerBase provides common infrastructure for a number of derived tracers.
 */
template <typename TracerTrait> class TracerBase {
public:
  using Data = typename TracerTrait::Data;
  using Integrator = typename TracerTrait::Integrator;
  using IntegrationStep = typename Integrator::IntegrationStep;
  using Recorder = typename TracerTrait::Recorder;

  TracerBase(Recorder *recorder)
      : data_{nullptr},
        recorder_(recorder), default_step_{0.01f}, num_iterations_{100} {
    assert(recorder_ != nullptr);
  }
  virtual ~TracerBase() = default;

  void SetData(const Data *data) { data_ = data; }
  const Data *GetData() const { return data_; }

  void SetIntegrationStep(const float_t step) { default_step_ = step; }
  float_t GetIntegrationStep() const { return default_step_; }

  void SetNumberOfIterations(const std::size_t n) { num_iterations_ = n; }
  std::size_t GetNumberOfIterations() const { return num_iterations_; }

protected:
  Recorder *GetRecorder() const { return recorder_; }

  std::vector<point_t> InitOutput(const std::size_t num_seeds) {
    recorder_->InitRecording(num_seeds, num_iterations_ / 4);
    std::vector<point_t> output_points;
    output_points.resize(num_seeds);
    return output_points;
  }

  IntegrationStep InitStep(std::size_t trace_id, const point_t &seed) {
    recorder_->InitTrace(trace_id, seed);
    return IntegrationStep{seed, default_step_};
  }

  IntegrationStep AdvectParticle(std::size_t trace_id,
                                 const Integrator &integrator,
                                 IntegrationStep &&step) {
    for (std::size_t iteration = 0; iteration < num_iterations_; ++iteration) {
      integrator.ComputeStep(step);
      if (step.HasFinished()) {
        recorder_->AppendTracePoint(trace_id, step.end_point);
        step.RestartAt(step.end_point);
      } else {
        step.end_point = step.start_point;
        break;
      }
    }
    return step;
  }

private:
  const Data *data_;
  float_t default_step_;
  std::size_t num_iterations_;
  Recorder *recorder_;
};
};

#endif
