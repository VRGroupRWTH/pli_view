#ifndef PLI_VIS_IO_VOLUME_IMPL_HPP_
#define PLI_VIS_IO_VOLUME_IMPL_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <utility>

#include <boost/multi_array.hpp>

namespace pli
{
// Implementation for MSA0309 type data (where all information is stored in a single volume, ZXY for scalars, ZWXY for vectors, ZWXY for tensors).
class io_volume_impl
{
public:
  static std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_scalar_dataset_bounds(
    const HighFive::File& file        , 
    const std::string&    dataset_path)
  {
    if (!file.isValid() || dataset_path.empty())
      return {{0, 0, 0}, {0, 0, 0}};
    auto unordered_size = file.getDataSet(dataset_path).getSpace().getDimensions();
    return {{0, 0, 0}, {unordered_size[1], unordered_size[2], unordered_size[0]}};
  }
  static std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> load_vector_dataset_bounds(
    const HighFive::File& file        , 
    const std::string&    dataset_path)
  {
    return load_tensor_dataset_bounds(file, dataset_path);
  }
  static std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> load_tensor_dataset_bounds(
    const HighFive::File& file        , 
    const std::string&    dataset_path)
  {
    if (!file.isValid() || dataset_path.empty())
      return {{0, 0, 0, 0}, {0, 0, 0, 0}};
    auto unordered_size = file.getDataSet(dataset_path).getSpace().getDimensions();
    return {{0, 0, 0, 0}, {unordered_size[2], unordered_size[3], unordered_size[0], unordered_size[1]}};
  }

  static boost::multi_array<float, 3> load_scalar_dataset(
    const HighFive::File&             file        , 
    const std::string&                dataset_path, 
    const std::array<std::size_t, 3>& offset      , 
    const std::array<std::size_t, 3>& size        , 
    const std::array<std::size_t, 3>& stride      , 
    bool                              normalize   )
  {
    if (!file.isValid() || dataset_path.empty())
      return boost::multi_array<float, 3>();

    boost::multi_array<float, 3> unordered_data;
    file
      .getDataSet(dataset_path)
      .select    ({offset[2], offset[0], offset[1]}, {size[2], size[0], size[1]}, std::vector<std::size_t>{stride[2], stride[0], stride[1]})
      .read      (unordered_data);

    // Convert ZXY to XYZ.
    boost::multi_array<float, 3> data(boost::extents[size[0]][size[1]][size[2]]);
    for (auto x = 0; x < size[0]; x++)
      for (auto y = 0; y < size[1]; y++)
        for (auto z = 0; z < size[2]; z++)
          data[x][y][z] = unordered_data[z][x][y];

    if (normalize && data.num_elements() > 0)
    {
      auto max_element = *std::max_element(data.data(), data.data() + data.num_elements());
      std::transform(data.data(), data.data() + data.num_elements(), data.data(), [max_element](const float& element)
      {
        return element / max_element;
      });
    }

    return data;
  }
  static boost::multi_array<float, 4> load_vector_dataset(
    const HighFive::File&             file        , 
    const std::string&                dataset_path, 
    const std::array<std::size_t, 3>& offset      , 
    const std::array<std::size_t, 3>& size        , 
    const std::array<std::size_t, 3>& stride      , 
    bool                              normalize   )
  {
    return load_tensor_dataset(file, dataset_path, offset, size, stride, normalize);
  }
  static boost::multi_array<float, 4> load_tensor_dataset(
    const HighFive::File&             file        , 
    const std::string&                dataset_path, 
    const std::array<std::size_t, 3>& offset      , 
    const std::array<std::size_t, 3>& size        , 
    const std::array<std::size_t, 3>& stride      , 
    bool                              normalize   )
  {
    if (!file.isValid() || dataset_path.empty())
      return boost::multi_array<float, 4>();

    auto dataset = file.getDataSet(dataset_path);
    auto count   = dataset.getSpace().getDimensions()[1];

    boost::multi_array<float, 4> unordered_data;
    dataset
      .select({offset[2], 0, offset[0], offset[1]}, {size[2], count, size[0], size[1]}, std::vector<std::size_t>{stride[2], 1, stride[0], stride[1]})
      .read  (unordered_data);

    // Convert ZVXY to XYZV.
    boost::multi_array<float, 4> data(boost::extents[size[0]][size[1]][size[2]][count]);
    for (auto x = 0; x < size[0]; x++)
      for (auto y = 0; y < size[1]; y++)
        for (auto z = 0; z < size[2]; z++)
          for (auto v = 0; v < count; v++)
            data[x][y][z][v] = unordered_data[z][v][x][y];

    if (normalize && data.num_elements() > 0)
    {
      auto max_element = *std::max_element(data.data(), data.data() + data.num_elements());
      std::transform(data.data(), data.data() + data.num_elements(), data.data(), [max_element](const float& element)
      {
        return element / max_element;
      });
    }

    return data;
  }
  
  static void save_scalar_dataset(
    HighFive::File&                     file        , 
    const std::string&                  dataset_path, 
    const std::array<std::size_t, 3>&   offset      , 
    const boost::multi_array<float, 3>& data        ) 
  {
    if (!file.isValid() || dataset_path.empty())
      return;

    auto shape = data.shape();

    // Convert XYZ to ZXY.
    boost::multi_array<float, 3> unordered_data(boost::extents[shape[2]][shape[0]][shape[1]]);
    for (auto x = 0; x < shape[0]; x++)
      for (auto y = 0; y < shape[1]; y++)
        for (auto z = 0; z < shape[2]; z++)
          unordered_data[z][x][y] = data[x][y][z];

    try
    {
      file
        .getDataSet   (dataset_path)
        .select       ({offset[2], offset[0], offset[1]}, {shape[2], shape[0], shape[1]})
        .write        (unordered_data);
    }
    catch (...)
    {
      file
        .createDataSet(dataset_path, HighFive::DataSpace({offset[2] + shape[2], offset[0] + shape[0], offset[1] + shape[1]}), HighFive::AtomicType<float>())
        .select       ({offset[2], offset[0], offset[1]}, {shape[2], shape[0], shape[1]})
        .write        (unordered_data);
    }

  } 
  static void save_vector_dataset(
    HighFive::File&                     file        , 
    const std::string&                  dataset_path, 
    const std::array<std::size_t, 3>&   offset      , 
    const boost::multi_array<float, 4>& data        )
  {
    save_tensor_dataset(file, dataset_path, offset, data);
  }
  static void save_tensor_dataset(
    HighFive::File&                     file        , 
    const std::string&                  dataset_path, 
    const std::array<std::size_t, 3>&   offset      , 
    const boost::multi_array<float, 4>& data        ) 
  {
    if (!file.isValid() || dataset_path.empty())
      return;

    auto shape = data.shape();

    // Convert XYZV to ZVXY.
    boost::multi_array<float, 4> unordered_data(boost::extents[shape[2]][shape[3]][shape[0]][shape[1]]);
    for (auto x = 0; x < shape[0]; x++)
      for (auto y = 0; y < shape[1]; y++)
        for (auto z = 0; z < shape[2]; z++)
          for (auto v = 0; v < shape[3]; v++)
            unordered_data[z][v][x][y] = data[x][y][z][v];

    try
    {
      file
        .getDataSet   (dataset_path)
        .select       ({offset[2], 0, offset[0], offset[1]}, {shape[2], shape[3], shape[0], shape[1]})
        .write        (unordered_data);
    }
    catch (...)
    {
      file
        .createDataSet(dataset_path, HighFive::DataSpace({offset[2] + shape[2], shape[3], offset[0] + shape[0], offset[1] + shape[1]}), HighFive::AtomicType<float>())
        .select       ({offset[2], 0, offset[0], offset[1]}, {shape[2], shape[3], shape[0], shape[1]})
        .write        (unordered_data);
    }
  }
};
}

#endif
