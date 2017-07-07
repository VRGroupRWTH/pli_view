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
#ifndef RAW_BINARY_READER_H
#define RAW_BINARY_READER_H

#include "base_types.hpp"
#include "cartesian_grid.hpp"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <vector>

namespace tangent {
class RawBinaryReader final {
public:
  RawBinaryReader() = default;
  ~RawBinaryReader() = default;

  std::unique_ptr<CartesianGrid>
  ReadFile(const std::string &file_name,
           const vector_t &spacing = {1.0f, 1.0f, 1.0f},
           const point_t &origin = {0.0f, 0.0f, 0.0f, 0.0f}) const {
    std::ifstream input_file{file_name.c_str(),
                             std::ifstream::in | std::ifstream::binary};
    if (!input_file.is_open())
      return nullptr;
    grid_dim_t dims{this->ReadGridDims(input_file)};
    if (input_file.fail())
      return nullptr;
    auto out_data = std::make_unique<CartesianGrid>(dims, spacing, origin);
    if (!this->ReadVectorData(input_file, out_data.get()))
      return nullptr;
    return out_data;
  }

private:
  tangent::grid_dim_t ReadGridDims(std::ifstream &file) const {
    std::uint32_t input_dims[3];
    file.read(reinterpret_cast<char *>(input_dims), 3 * sizeof(std::uint32_t));
    return grid_dim_t{input_dims[0], input_dims[1], input_dims[2]};
  }

  bool ReadVectorData(std::ifstream &input_file, CartesianGrid *data) const {
    auto dims = data->GetDims();
    std::size_t num_vectors = dims[0] * dims[1] * dims[2];
    std::size_t num_bytes = num_vectors * 3 * sizeof(float);
    // NOTE: This only works if the 32-bit data in the file matches the 32-bit
    // runtime representation of tangent!
    assert(sizeof(float) == sizeof(tangent::float_t));
    std::vector<tangent::vector_t> input_data(num_vectors);
    input_file.read(reinterpret_cast<char *>(&input_data[0]), num_bytes);
    if (ReadFailed(input_file) || !ReadFileToTheEnd(input_file))
      return false;
    data->Assign(std::move(input_data));
    return true;
  }

  bool ReadFailed(std::ifstream &f) const { return f.fail(); }
  bool ReadFileToTheEnd(std::ifstream &f) const { return f.peek() == EOF; }
};
}
#endif