#ifndef PLI_IO_HDF5_IO_2_HPP_
#define PLI_IO_HDF5_IO_2_HPP_

#define H5_USE_BOOST_MULTI_ARRAY

#include <array>
#include <functional>

#include <boost/multi_array.hpp>

#include <third_party/highfive/H5DataSet.hpp>
#include <third_party/highfive/H5File.hpp>

#include "hdf5_io_base.hpp"

namespace pli
{
// For MSA0309 style data (Volume. ZXY for scalars, ZVXY for vectors, ZTXY for tensors).
class hdf5_io_2 : public hdf5_io_base
{
public:
  explicit hdf5_io_2(
    std::string filepath,
    std::string attribute_path_vector_spacing   = std::string(),
    std::string attribute_path_block_size       = std::string(),
    std::string dataset_path_mask               = std::string(),
    std::string dataset_path_transmittance      = std::string(),
    std::string dataset_path_retardation        = std::string(),
    std::string dataset_path_fiber_direction    = std::string(),
    std::string dataset_path_fiber_inclination  = std::string(),
    std::string dataset_path_fiber_unit_vectors = std::string(),
    std::string dataset_path_fiber_distribution = std::string())
    : hdf5_io_base(
      filepath                       ,
      attribute_path_vector_spacing  ,
      attribute_path_block_size      ,
      dataset_path_mask              ,
      dataset_path_transmittance     ,
      dataset_path_retardation       ,
      dataset_path_fiber_direction   ,
      dataset_path_fiber_inclination ,
      dataset_path_fiber_unit_vectors,
      dataset_path_fiber_distribution)
  {

  }
  
private:
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_scalar_dataset_bounds(std::string dataset_path) override
  {
    if (!file_.isValid() || dataset_path.empty())
      return {{0, 0, 0}, {0, 0, 0}};
    auto misordered_size = file_.getDataSet(dataset_path).getSpace().getDimensions();
    return {{0, 0, 0}, {misordered_size[1], misordered_size[2], misordered_size[0]}};
  }
  std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> load_vector_dataset_bounds(std::string dataset_path) override
  {
    return load_tensor_dataset_bounds(dataset_path);
  }
  std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> load_tensor_dataset_bounds(std::string dataset_path) override
  {
    if (!file_.isValid() || dataset_path.empty())
      return {{0, 0, 0, 0}, {0, 0, 0, 0}};
    auto misordered_size = file_.getDataSet(dataset_path).getSpace().getDimensions();
    return {{0, 0, 0, 0}, {misordered_size[2], misordered_size[3], misordered_size[0], misordered_size[1]}};
  }

  boost::multi_array<float, 3> load_scalar_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride, bool normalize) override
  {
    if (!file_.isValid() || dataset_path.empty())
      return boost::multi_array<float, 3>();

    boost::multi_array<float, 3> misordered_data;
    file_
      .getDataSet(dataset_path)
      .select    ({offset[2], offset[0], offset[1]}, {size[2], size[0], size[1]}, std::vector<std::size_t>{stride[2], stride[0], stride[1]})
      .read      (misordered_data);

    // Convert ZXY to XYZ.
    boost::multi_array<float, 3> data(boost::extents[size[0]][size[1]][size[2]]);
    for (auto x = 0; x < size[0]; x++)
      for (auto y = 0; y < size[1]; y++)
        for (auto z = 0; z < size[2]; z++)
          data[x][y][z] = misordered_data[z][x][y];

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
  boost::multi_array<float, 4> load_vector_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride, bool normalize) override
  {
    return load_tensor_dataset(dataset_path, offset, size, stride, normalize);
  }
  boost::multi_array<float, 4> load_tensor_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride, bool normalize) override
  {
    if (!file_.isValid() || dataset_path.empty())
      return boost::multi_array<float, 4>();

    auto dataset = file_.getDataSet(dataset_path);
    auto count   = dataset.getSpace().getDimensions()[1];

    boost::multi_array<float, 4> misordered_data;
    dataset
      .select({offset[2], 0, offset[0], offset[1]}, {size[2], count, size[0], size[1]}, std::vector<std::size_t>{stride[2], 1, stride[0], stride[1]})
      .read  (misordered_data);

    // Convert ZVXY to XYZV.
    boost::multi_array<float, 4> data(boost::extents[size[0]][size[1]][size[2]][count]);
    for (auto x = 0; x < size[0]; x++)
      for (auto y = 0; y < size[1]; y++)
        for (auto z = 0; z < size[2]; z++)
          for (auto v = 0; v < count; v++)
            data[x][y][z][v] = misordered_data[z][v][x][y];

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
  
  void save_scalar_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data) override 
  {
    if (!file_.isValid() || dataset_path.empty())
      return;

    auto shape = data.shape();

    // Convert XYZ to ZXY.
    boost::multi_array<float, 3> misordered_data(boost::extents[shape[2]][shape[0]][shape[1]]);
    for (auto x = 0; x < shape[0]; x++)
      for (auto y = 0; y < shape[1]; y++)
        for (auto z = 0; z < shape[2]; z++)
          misordered_data[z][x][y] = data[x][y][z];

    try
    {
      file_
        .getDataSet   (dataset_path)
        .select       ({offset[2], offset[0], offset[1]}, {shape[2], shape[0], shape[1]})
        .write        (misordered_data);
    }
    catch (...)
    {
      file_
        .createDataSet(dataset_path, HighFive::DataSpace({offset[2] + shape[2], offset[0] + shape[0], offset[1] + shape[1]}), HighFive::AtomicType<float>())
        .select       ({offset[2], offset[0], offset[1]}, {shape[2], shape[0], shape[1]})
        .write        (misordered_data);
    }

  } 
  void save_vector_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 4>& data) override
  {
    save_tensor_dataset(dataset_path, offset, data);
  }
  void save_tensor_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 4>& data) override 
  {
    if (!file_.isValid() || dataset_path.empty())
      return;

    auto shape = data.shape();

    // Convert XYZV to ZVXY.
    boost::multi_array<float, 4> misordered_data(boost::extents[shape[2]][shape[3]][shape[0]][shape[1]]);
    for (auto x = 0; x < shape[0]; x++)
      for (auto y = 0; y < shape[1]; y++)
        for (auto z = 0; z < shape[2]; z++)
          for (auto v = 0; v < shape[3]; v++)
            misordered_data[z][v][x][y] = data[x][y][z][v];

    try
    {
      file_
        .getDataSet   (dataset_path)
        .select       ({offset[2], 0, offset[0], offset[1]}, {shape[2], shape[3], shape[0], shape[1]})
        .write        (misordered_data);
    }
    catch (...)
    {
      file_
        .createDataSet(dataset_path, HighFive::DataSpace({offset[2] + shape[2], shape[3], offset[0] + shape[0], offset[1] + shape[1]}), HighFive::AtomicType<float>())
        .select       ({offset[2], 0, offset[0], offset[1]}, {shape[2], shape[3], shape[0], shape[1]})
        .write        (misordered_data);
    }
  }
};
}

#endif
