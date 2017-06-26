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
#ifndef TBB_POS_TRACER_H
#define TBB_POS_TRACER_H

#include <tangent-base/base_types.hpp>
#include <tangent-base/tracer_base.hpp>

#include <vector>

#include <tbb/tbb.h>

namespace tangent {
/**
 * TBBPOSTracer uses Intel Threading Building Blocks (TBB) for a parallel
 * particle integration based on the "parallelize-over-seed" (POS) strategy.
 */
template <typename TracerTrait>
class TBBPOSTracer final : public TracerBase<TracerTrait> {
public:
  using Data = typename TracerTrait::Data;
  using Integrator = typename TracerTrait::Integrator;
  using IntegrationStep = typename Integrator::IntegrationStep;
  using Recorder = typename TracerTrait::Recorder;

  TBBPOSTracer(Recorder *recorder) : TracerBase<TracerTrait>(recorder){};
  ~TBBPOSTracer() = default;

  std::vector<point_t> TraceSeeds(const std::vector<point_t> &seeds) {
    const std::int64_t num_seeds = static_cast<std::int64_t>(seeds.size());
    auto output_points = this->InitOutput(num_seeds);
    TraceFunctor advect_seeds{this, seeds, output_points};
    tbb::parallel_for(tbb::blocked_range<int>(0, static_cast<int>(num_seeds)),
                      advect_seeds);
    return output_points;
  }

protected:
  struct TraceFunctor {
    TBBPOSTracer<TracerTrait> *tracer_;
    const std::vector<point_t> &seeds_;
    std::vector<point_t> &output_points_;

    TraceFunctor(TBBPOSTracer<TracerTrait> *tracer,
                 const std::vector<point_t> &seeds,
                 std::vector<point_t> &output_points)
        : tracer_{tracer}, seeds_{seeds}, output_points_{output_points} {}

    void operator()(const tbb::blocked_range<int> &range) const {
      Integrator integrator(tracer_->GetData());
      for (int seed_id = range.begin(); seed_id != range.end(); ++seed_id) {
        auto result = tracer_->AdvectParticle(
            seed_id, integrator, tracer_->InitStep(seed_id, seeds_[seed_id]));
        output_points_[seed_id] = result.end_point;
      }
    }
  };
  friend TraceFunctor;

private:
};
}

#endif