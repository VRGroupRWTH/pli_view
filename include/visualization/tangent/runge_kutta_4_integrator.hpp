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
#ifndef TANGENTBASE_RUNGE_KUTTA_4_INTEGRATOR_HPP
#define TANGENTBASE_RUNGE_KUTTA_4_INTEGRATOR_HPP

#include "base_types.hpp"

namespace tangent {

/**
 * Vanilla Runge Kutta 4 scheme w/o step size adaption (for now).
 */
template <typename InterpolatorTrait> class RungeKutta4Integrator final {
public:
  using Interpolator = typename InterpolatorTrait::Interpolator;
  using Data = typename InterpolatorTrait::Data;

  enum StepState {
    STEP_STAGE_0 = 0,
    STEP_STAGE_1 = 1,
    STEP_STAGE_2 = 2,
    STEP_STAGE_3 = 3,
    STEP_FINISHED = 4
  };

  /**
   * Helper class to facilitate pre-empted integration across multiple data
   * sets. All necessary information from a step is stored in an instance of
   * IntegrationStep. Hence, should the step fail at any stage during the
   * integration - i.e. in case the  current sample pos lies outside the current
   * data - it can be resumed later on, specifically after having provided the
   * correct data set. This ensures that an integration across multiple blocks
   * performs exactly the same computations as an integration within a single
   * block.
   */
  struct IntegrationStep final {
    IntegrationStep(const tangent::point_t &start_pt,
                    const tangent::float_t step)
        : start_point(start_pt), end_point(start_pt),
          current_k({0.0f, 0.0f, 0.0f}), k_sum({0.0f, 0.0f, 0.0f}),
          time_step(step), next_stage(STEP_STAGE_0) {}
    ~IntegrationStep() = default;

    void RestartAt(point_t &start) {
      start_point = start;
      k_sum = {0.0f, 0.0f, 0.0f};
      next_stage = STEP_STAGE_0;
    }

    bool HasFinished() const { return next_stage == STEP_FINISHED; }

    tangent::point_t start_point;
    tangent::point_t end_point;
    tangent::vector_t current_k;
    tangent::vector_t k_sum;
    tangent::float_t time_step;
    stage_id_t next_stage;
  };

  RungeKutta4Integrator(const Data &data)
      : interpolator_{data}, stage_factors_({0.0f, 0.5f, 0.5f, 1.0f}),
        result_factors_({1.0f, 2.0f, 2.0f, 1.0f}) {}
  ~RungeKutta4Integrator() = default;

  /**
   * Perform a single  RK4 step. In case this is successful, step_data will
   * contain a valid end_point and its next_stage attribute will be set to
   * STEP_FINISHED. In case something goes wrong, intermediate values will be
   * stored in step_data, in order to be able to resume integration.
   * Specifically, step_data.next_stage will indicate the next integration stage
   * that has to be performed by the integrator.
   * Note: For use in repeated integration, it is the client's responsibility to
   * correctly init step_data after each iteration. The STEP_FINISHED state is
   * specifically included in order to separate a failed first stage from a
   * fully completed step.
   */
  void Step(IntegrationStep &step_data) const {
    point_t current_point;
    vector_t current_vector;
    for (stage_id_t stage = 0; stage < 4; ++stage) {
      current_point = this->GetStateSamplePosition(stage, step_data);
      if (!this->Interpolate(current_point, current_vector))
        return;
      this->UpdateStepData(stage, current_vector, step_data);
    }
    this->FinalizeStep(step_data);
  }

  /**
   * Resume a previously started integration step with the given parameters and
   * partial information stored in step_data.
   */
  void ResumeStep(IntegrationStep &step_data) const {
    point_t current_point;
    vector_t current_vector;
    for (stage_id_t stage = 0; stage < 4; ++stage) {
      if (step_data.next_stage > stage)
        continue;
      current_point = this->GetStateSamplePosition(stage, step_data);
      if (!this->Interpolate(current_point, current_vector))
        return;
      this->UpdateStepData(stage, current_vector, step_data);
    }
    this->FinalizeStep(step_data);
  }

protected:
  point_t GetStateSamplePosition(stage_id_t stage,
                                 IntegrationStep &step_data) const {
    point_t sample_position = tangent::AddWeightedVectorToPoint(
        step_data.start_point, stage_factors_[stage], step_data.current_k);
    sample_position[3] += stage_factors_[stage] * step_data.time_step;
    return sample_position;
  }

  bool Interpolate(const point_t &sample_point, vector_t &vector) const {
    if (!interpolator_.IsInside(sample_point))
      return false;
    vector = interpolator_.Interpolate(sample_point);
    return true;
  }

  void UpdateStepData(stage_id_t stage, const vector_t &current_vector,
                      IntegrationStep &step_data) const {
    step_data.current_k =
        tangent::ScaleVector(current_vector, step_data.time_step);
    /*
    Note: first sum up the k values only. Do not incrementally sum up the end
    position. The latter options, while more compact to write, is numerically
    less stable.
    */
    step_data.k_sum = tangent::AddWeightedVector(
        step_data.k_sum, result_factors_[stage], step_data.current_k);
    step_data.next_stage = stage + 1;
  }

  void FinalizeStep(IntegrationStep &step_data) const {
    step_data.end_point = tangent::AddWeightedVectorToPoint(
        step_data.start_point, (1.0f / 6.0f), step_data.k_sum);
    step_data.end_point[3] = step_data.start_point[3] + step_data.time_step;
  }

private:
  Interpolator interpolator_;
  std::array<float_t, 4> stage_factors_;
  std::array<float_t, 4> result_factors_;
};
}

#endif // Include guard.
