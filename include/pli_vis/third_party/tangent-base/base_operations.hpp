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
#ifndef TANGENT_BASE_BASE_OPERATIONS_HPP
#define TANGENT_BASE_BASE_OPERATIONS_HPP

#include "base_types.hpp"
#include <cmath>

namespace tangent {

inline vector_t Interpolate1D(const vector_t &v1, const vector_t &v2,
                              const float_t &weight) {
  return vector_t{{(1.0f - weight) * v1[0] + weight * v2[0],
                   (1.0f - weight) * v1[1] + weight * v2[1],
                   (1.0f - weight) * v1[2] + weight * v2[2]}};
}

inline vector_t AddWeightedVector(const vector_t &v, const float_t &f,
                                  const vector_t &u) {
  return vector_t{{v[0] + f * u[0], v[1] + f * u[1], v[2] + f * u[2]}};
}

inline point_t AddWeightedVectorToPoint(const point_t &p, const float_t &f,
                                        const vector_t &v) {
  return point_t{{p[0] + f * v[0], p[1] + f * v[1], p[2] + f * v[2], p[3]}};
}

inline vector_t ScaleVector(const vector_t &v, const float_t &f) {
  return vector_t{{v[0] * f, v[1] * f, v[2] * f}};
}

template <std::size_t array_size, std::size_t num_components>
inline float_t Length(const std::array<float_t, array_size> &a) {
  float_t l = 0;
  for (std::size_t i = 0; i < num_components; ++i)
    l += a[i] * a[i];
  return std::sqrt(l);
}

inline float_t Length(const point_t &p) { return Length<4, 3>(p); }

inline float_t Length(const vector_t &v) { return Length<3, 3>(v); }
}

#endif // Include guard.
