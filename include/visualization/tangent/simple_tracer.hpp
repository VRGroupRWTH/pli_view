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
#ifndef TANGENT_BASE_SIMPLE_TRACER_HPP
#define TANGENT_BASE_SIMPLE_TRACER_HPP

#include "base_types.hpp"
#include <vector>

namespace tangent {
template <typename TracerTrait> class SimpleTracer final {
public:
  using Data = typename TracerTrait::Data;
  using Integrator = typename TracerTrait::Integrator;
  using IntegrationStep = typename Integrator::IntegrationStep;
  using Recorder = typename TracerTrait::Recorder;

  SimpleTracer() : data_(nullptr), default_step_(0.01f), num_iterations_{ 100 } {}
  ~SimpleTracer() = default;

  void SetData(Data *data) { data_ = data; }

  void SetIntegrationStep(const float_t step){ default_step_ = step;  }

  void SetNumberOfIterations(const std::size_t n) { num_iterations_ = n; }

  std::vector<point_t> TraceSeeds(const std::vector<point_t> &seeds) {
    const auto num_seeds = seeds.size();
    auto output_points = this->InitOutput(num_seeds);
    Integrator integrator(*data_);
    for (std::size_t seed_id = 0; seed_id < num_seeds; ++seed_id){
      auto result = this->AdvectParticle(seed_id, integrator, this->InitStep(seed_id, seeds[seed_id]));
      output_points[seed_id] = result.end_point;
    }
    return output_points;
  }

  Recorder &GetRecorder() { return recorder_; }

protected:
  std::vector<point_t> InitOutput(const std::size_t num_seeds) {
    recorder_.InitRecording(num_seeds, 100);
    std::vector<point_t> output_points;
    output_points.resize(num_seeds);
    return output_points;
  }

  IntegrationStep InitStep(std::size_t trace_id, const point_t &seed){
    recorder_.InitTrace(trace_id, seed);
    return Integrator::IntegrationStep{ seed, default_step_ };
  }

  IntegrationStep AdvectParticle(std::size_t trace_id, const Integrator &integrator, IntegrationStep &&step) {
    for (std::size_t iteration = 0; iteration < num_iterations_; ++iteration) {
      integrator.Step(step);
      if (step.HasFinished()) {
        recorder_.AppendTracePoint(trace_id, step.end_point);
        step.RestartAt(step.end_point);
      }
      else {
        step.end_point = step.start_point;
        break;
      }
    }
    return step;
  }

private:
  Data *data_;
  float_t default_step_; 
  std::size_t num_iterations_;
  Recorder recorder_;
};
}

#endif // Include guard.
