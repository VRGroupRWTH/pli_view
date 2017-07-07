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
#ifndef TANGENTBASE_CARTESIANGRID_H
#define TANGENTBASE_CARTESIANGRID_H

#include <array>
#include <cassert>
#include <vector>

#include <tangent-base/base_types.hpp>

namespace tangent {

/**
 * straightforward implementation of a vector field represented on a Cartesian
 * grid. Note that this only implements the data handling itself, i.e. no
 * higher-level functionality, e.g. interpolation, is defined here.
 */
class CartesianGrid final {
public:
  /**
   * create a cartesian grid based on dims, cell spacing, and an (optional)
   * origin which defaults to 0.
   */
  explicit CartesianGrid(const grid_dim_t &dims, const vector_t &spacing,
                         const point_t &origin = {{0.0f, 0.0f, 0.0f, 0.0f}})
      : dims_{dims}, spacing_{spacing}, origin_{origin} {
    for (std::size_t d = 0; d < 3; ++d) {
      box_[2 * d] = origin_[d];
      box_[2 * d + 1] = origin_[d] + (dims_[d] - 1) * spacing_[d];
    }
    data_.resize(dims[0] * dims[1] * dims[2]);
  }
  /**
   * create a cartesian grid based on dims and a bounding box. Spacing
   * will automatically be computed according to these specs.
   */
  explicit CartesianGrid(const grid_dim_t &dims, const box_t box)
      : dims_(dims), origin_({{box[0], box[2], box[4], 0.0f}}), box_(box) {
    for (std::size_t d = 0; d < 3; ++d)
      spacing_[d] = (box[2 * d + 1] - box[2 * d]) / (dims[d] - 1);
    data_.resize(dims[0] * dims[1] * dims[2]);
  }

  ~CartesianGrid() = default;

  /**
   * flood fill the data with the given value
   */
  void Assign(const vector_t &value) {
    data_.assign(dims_[0] * dims_[1] * dims_[2], value);
  }

  void Assign(std::vector<vector_t> &&input_vals) {
    assert(input_vals.size() == dims_[0] * dims_[1] * dims_[2]);
    data_ = input_vals;
  }

  std::size_t GetNumberOfPoints() const { return data_.size(); }

  grid_dim_t GetDims() const { return dims_; }

  point_t GetOrigin() const { return origin_; }

  vector_t GetSpacing() const { return spacing_; }

  box_t GetBounds() const { return box_; }

  vector_t GetVectorValue(const point_id_t ptId) const {
    assert(ptId < data_.size());
    return data_[ptId];
  }

  /**
   * Get pointer into the underlying data representation. It is safe to assume
   * that vector data will be handled as contiguous chunk of interleaved data
   * tuples of the form [u,v,w,u,v,w,...]
   */
  vector_t *GetVectorPointer(const point_id_t ptId) {
    assert(ptId < data_.size());
    return &(data_[ptId]);
  }

protected:
private:
  grid_dim_t dims_;
  vector_t spacing_;
  point_t origin_;
  box_t box_;

  std::vector<tangent::vector_t> data_;
};
};

#endif
