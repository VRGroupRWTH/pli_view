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

#include "tracer_base.hpp"
#include "base_types.hpp"
#include <vector>

namespace tangent {
template <typename TracerTrait>
class SimpleTracer final : public TracerBase<TracerTrait> {
public:
  using Data = typename TracerTrait::Data;
  using Integrator = typename TracerTrait::Integrator;
  using IntegrationStep = typename Integrator::IntegrationStep;
  using Recorder = typename TracerTrait::Recorder;

  SimpleTracer(Recorder *recorder) : TracerBase<TracerTrait>(recorder) {}
  ~SimpleTracer() override = default;

  std::vector<point_t> TraceSeeds(const std::vector<point_t> &seeds) {
    const auto num_seeds = seeds.size();
    auto output_points = this->InitOutput(num_seeds);
    Integrator integrator(this->GetData());
    for (std::size_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
      auto result = this->AdvectParticle(
          seed_id, integrator, this->InitStep(seed_id, seeds[seed_id]));
      output_points[seed_id] = result.end_point;
    }
    return output_points;
  }

protected:
};
}

#endif // Include guard.
