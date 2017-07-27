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
#ifndef TANGENTBASE_ANALYTICORBITINTERPOL_H
#define TANGENTBASE_ANALYTICORBITINTERPOL_H

#include "base_operations.hpp"
#include "base_types.hpp"
#include "cartesian_grid.hpp"

namespace tangent {

class AnalyticOrbitInterpolator final {
public:
  AnalyticOrbitInterpolator() = default;
  AnalyticOrbitInterpolator(const CartesianGrid *data) {}
  ~AnalyticOrbitInterpolator() = default;

  bool IsInside(const point_t &point) const { return true; }

  vector_t Interpolate(const point_t &pt) const {
    float_t oneDivLen = 1.0f / Length(pt);
    vector_t normVec = {pt[0] * oneDivLen, pt[1] * oneDivLen,
                        pt[2] * oneDivLen};
    return vector_t({-normVec[1], normVec[0], 0.0f});
  }

private:
};
};

#endif