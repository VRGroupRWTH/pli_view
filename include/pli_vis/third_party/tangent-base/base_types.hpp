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
#ifndef TANGENTBASE_TYPES_H
#define TANGENTBASE_TYPES_H

#include <array>
#include <cstdint>

namespace tangent {
using float_t = float;
using point_id_t = std::size_t;
using stage_id_t = std::uint32_t;

using point_t = std::array<float_t, 4>;
using vector_t = std::array<float_t, 3>;

using weight_t = std::array<float_t, 3>;

using grid_dim_t = std::array<std::size_t, 3>;
using grid_coords_t = std::array<std::size_t, 3>;

/**
 * axis aligned bounding box in the format {min_x, max_x, min_y, max_y, min_z,
 * max_z}
 */
using box_t = std::array<float_t, 6>;

/**
 * keep track of the eight point ids that designate a voxel's vertices
 */
using cell_pointids_t = std::array<std::size_t, 8>;

/**
 * keep track of the 8 vector values attached to a voxel's vertices
 */
using cell_vectors_t = std::array<vector_t, 8>;
};

#endif