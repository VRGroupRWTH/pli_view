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
// (at your opointion) any later version.
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

#include "cartesian_grid.hpp"

#include <cassert>

namespace tangent {
class CartesianLocator {
public:
  CartesianLocator(const CartesianGrid &grid)
      : gridBox_(grid.GetBounds()), gridDims_(grid.GetDims()),
        numGridPoints_(gridDims_[0] * gridDims_[1] * gridDims_[2]),
        gridOrigin_(grid.GetOrigin()), gridSpacing_(grid.GetSpacing()) {}
  ~CartesianLocator() = default;

  /**
  * Check whether or not the given point (in world coords) is inside the box
  * covered by the underlying grid. Note that the lower boundary is considered
  * part of the grid whereas the upper boundary is not. Hence,
  * isPointInsideGrid(origin)== true, but
  * isPointInsideGrid({box[1], box[3], box[5]} == false.
  */
  bool IsInsideGrid(const point_t &point) const {
    return (point[0] >= gridBox_[0] && point[0] < gridBox_[1]) &&
           (point[1] >= gridBox_[2] && point[1] < gridBox_[3]) &&
           (point[2] >= gridBox_[4] && point[2] < gridBox_[5]);
  }

  /**
   * Get the grid coords of the base point (i.e. the lower, left, back vertex
   * id) of the voxel containing pt iff pt is inside the grid. Note that the
   * result is undefined if pt is not inside the grid's BB. Use isInsideGrid if
   * in doubt.
   */
  grid_coords_t GetGridCoords(const point_t &point) const {
    assert(this->IsInsideGrid(point));
    return grid_coords_t({static_cast<point_id_t>(std::floor(
                              (point[0] - gridOrigin_[0]) / gridSpacing_[0])),
                          static_cast<point_id_t>(std::floor(
                              (point[1] - gridOrigin_[1]) / gridSpacing_[1])),
                          static_cast<point_id_t>(std::floor(
                              (point[2] - gridOrigin_[2]) / gridSpacing_[2]))});
  }

  /**
   * Get the grid coords for a given, valid point id of a grid vertex.
   */
  grid_coords_t GetGridCoords(const point_id_t ptId) const {
    assert(ptId < numGridPoints_);
    return tangent::grid_coords_t({(ptId % gridDims_[0]),
                                   (ptId / gridDims_[0]) % gridDims_[1],
                                   (ptId / gridDims_[0]) / gridDims_[1]});
  }

  /**
   * Get the parametric coordinates of the given point inside the voxel
   * which contains it. Note that the result is undefined if pt is not inside
   * the grid's BB. Use isInside if in doubt.
   */
  weight_t GetParametricCoords(const point_t &point) const {
    point_t basePt = this->GetPointCoords(this->GetGridCoords(point));
    return weight_t({(point[0] - basePt[0]) / gridSpacing_[0],
                     (point[1] - basePt[1]) / gridSpacing_[1],
                     (point[2] - basePt[2]) / gridSpacing_[2]});
  }

  /**
   * Return the 3D position of the grid point designated by the given grid
   * coords. Note that this will return invalid points if gridCoords is outside
   * the grid's dims in any dimension.
   */
  point_t GetPointCoords(const grid_coords_t &gridCoords) const {
    assert(gridCoords[0] < gridDims_[0] && gridCoords[1] < gridDims_[1] &&
           gridCoords[2] < gridDims_[2]);
    return point_t({gridOrigin_[0] + gridCoords[0] * gridSpacing_[0],
                    gridOrigin_[1] + gridCoords[1] * gridSpacing_[1],
                    gridOrigin_[2] + gridCoords[2] * gridSpacing_[2]});
  }

  /**
   * Convert grid coords to point id
   */
  point_id_t GetPointId(const grid_coords_t &gridCoords) const {
    assert(gridCoords[0] < gridDims_[0] && gridCoords[1] < gridDims_[1] &&
           gridCoords[2] < gridDims_[2]);
    return gridCoords[0] +
           gridDims_[0] * (gridCoords[1] + gridDims_[1] * gridCoords[2]);
  }

  /**
   * Given grid coords of a valid grid point that has a voxel to its right,
   * upper, front, retrieve the grid point ids which define that voxel Note that
   * the behavior is undefined if pt lies outside the grid's bounding box.
   */
  cell_pointids_t GetCellPointIds(const grid_coords_t &gridCoords) const {
    point_id_t ptId = this->GetPointId(gridCoords);
    return cell_pointids_t({ptId, ptId + 1, ptId + gridDims_[0],
                            ptId + gridDims_[0] + 1,
                            ptId + gridDims_[0] * gridDims_[1],
                            ptId + gridDims_[0] * gridDims_[1] + 1,
                            ptId + (gridDims_[0] + 1) * gridDims_[1],
                            ptId + (gridDims_[0] + 1) * gridDims_[1] + 1});
  }

private:
  const box_t gridBox_;
  const grid_dim_t gridDims_;
  const unsigned long long numGridPoints_;
  const point_t gridOrigin_;
  const vector_t gridSpacing_;
};
};