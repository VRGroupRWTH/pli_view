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
#ifndef MEASUREMENT_BASE_H
#define MEASUREMENT_BASE_H

#include <chrono>
#include <iostream>
#include <string>

namespace tangent {

class MeasurementBase {
public:
  MeasurementBase(const std::string &label) : label_{label} {}
  virtual ~MeasurementBase() = default;

  const std::string GetLabel() const { return label_; }

  virtual std::ostream &WriteData(std::ostream &out_stream) const = 0;

protected:
private:
  std::string label_;
};
};

#endif