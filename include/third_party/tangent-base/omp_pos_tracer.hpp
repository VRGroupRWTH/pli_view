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
#ifndef OMP_POS_TRACER_H
#define OMP_POS_TRACER_H

#include "base_types.hpp"
#include "tracer_base.hpp"

#include <vector>

namespace tangent {
/**
 * OmpPOSTracer uses OpenMP for a parallel particle integration based on the
 * "parallelize-over-seed" (POS) strategy.
 */
template <typename TracerTrait>
class OmpPOSTracer final : public TracerBase<TracerTrait> {
public:
  using Data = typename TracerTrait::Data;
  using Integrator = typename TracerTrait::Integrator;
  using IntegrationStep = typename Integrator::IntegrationStep;
  using Recorder = typename TracerTrait::Recorder;

  OmpPOSTracer(Recorder *recorder) : TracerBase<TracerTrait>(recorder){};
  ~OmpPOSTracer() = default;

  std::vector<point_t> TraceSeeds(const std::vector<point_t> &seeds) {
    const std::int64_t num_seeds = static_cast<std::int64_t>(seeds.size());
    auto output_points = this->InitOutput(num_seeds);

#pragma omp parallel shared(output_points)
    {
      Integrator integrator(this->GetData());
#pragma omp for schedule(guided)
      for (std::int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
        auto result = this->AdvectParticle(
            seed_id, integrator, this->InitStep(seed_id, seeds[seed_id]));
        output_points[seed_id] = result.end_point;
      }
    }
    return output_points;
  }

protected:
private:
};
}

#endif