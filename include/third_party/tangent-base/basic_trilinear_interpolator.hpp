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
#ifndef TANGENTBASE_BASICTRILINEARINTERPOLATOR_H
#define TANGENTBASE_BASICTRILINEARINTERPOLATOR_H

#include "base_operations.hpp"
#include "base_types.hpp"
#include "cartesian_grid.hpp"
#include "cartesian_locator.hpp"
#include <cassert>

namespace tangent {

/**
 * Implementation of a very basic trilinear interpolator which performs spatial
 * interpolation and uses no optimizations whatsoever.
 */
class BasicTrilinearInterpolator final {
public:
  /**
   * Create an interpolator based on the Cartesian grid passed in data
   */
  BasicTrilinearInterpolator(const CartesianGrid *data)
      : data_{data}, locator_{data} {
    assert(data_ != nullptr);
  }
  ~BasicTrilinearInterpolator() = default;

  void SetData(const CartesianGrid *data) {
    assert(data != nullptr);
    locator_.SetData(data);
    data_ = data;
  }

  /**
   * Check whether the given position is inside the underlying data's bounding
   * box
   */
  bool IsInside(const point_t &point) const {
    return locator_.IsInsideGrid(point);
  }

  /**
   * Perform trilinear interpolation by querying the underling Cartesian grid.
   */
  vector_t Interpolate(const point_t &pt) const {
    assert(locator_.IsInsideGrid(pt));

    auto cellPoints = locator_.GetCellPointIds(locator_.GetGridCoords(pt));
    cell_vectors_t vectorVals = this->GetVectorValues(cellPoints);
    weight_t weights = locator_.GetParametricCoords(pt);

    vectorVals = this->FoldAlongAxis<4>(vectorVals, weights[0]);
    vectorVals = this->FoldAlongAxis<2>(vectorVals, weights[1]);
    return this->FoldAlongAxis<1>(vectorVals, weights[2])[0];
  }

protected:
  cell_vectors_t GetVectorValues(const cell_pointids_t &cellPoints) const {
    cell_vectors_t vectorVals;
    for (std::size_t p = 0; p < 8; ++p)
      vectorVals[p] = data_->GetVectorValue(cellPoints[p]);
    return vectorVals;
  }

  template <std::size_t NUM_DIMS>
  cell_vectors_t FoldAlongAxis(const std::array<vector_t, 8> &values,
                               const float_t weight) const {
    cell_vectors_t outVals;
    for (std::size_t i = 0; i < NUM_DIMS; ++i)
      outVals[i] =
          tangent::Interpolate1D(values[2 * i], values[2 * i + 1], weight);
    return outVals;
  }

private:
  const CartesianGrid *data_;
  CartesianLocator locator_;
};
};

#endif
