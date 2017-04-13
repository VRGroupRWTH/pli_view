#ifndef PLI_IO_HDF5_IO_HPP_
#define PLI_IO_HDF5_IO_HPP_

#define H5_USE_BOOST_MULTI_ARRAY

#include <array>
#include <functional>
#include <iostream>

#include <boost/algorithm/string/replace.hpp>
#include <boost/format.hpp>
#include <boost/multi_array.hpp>

#include <third_party/highfive/H5DataSet.hpp>
#include <third_party/highfive/H5File.hpp>

#include "hdf5_io_base.hpp"

namespace pli
{
// For Vervet1818 style data (each slice is a separate dataset).
class hdf5_io : public hdf5_io_base
{
public:
  explicit hdf5_io(
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
  boost::multi_array<float, 3> load_scalar_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, bool normalize) override
  {
    boost::multi_array<float, 3> data(boost::extents[size[0]][size[1]][size[2]]);
  
    if (!file_.isValid() || dataset_path.empty())
      return data;

    dataset_path = boost::replace_first_copy(dataset_path, "%Slice%", "%04d");

    for (auto z = 0; z < size[2]; z++)
    {
      boost::multi_array<float, 2> slice_data;
      file_
        .getDataSet((boost::format(dataset_path) % (z + offset[2])).str())
        .select    ({offset[0], offset[1]}, {size[0], size[1]})
        .read      (slice_data);
      data[boost::indices[index_range()][index_range()][z]] = slice_data;
    }

    if (normalize)
    {
      auto max_element = *std::max_element(data.data(), data.data() + data.num_elements());
      std::transform(data.data(), data.data() + data.num_elements(), data.data(), [max_element](const float& element)
      {
        return element / max_element;
      });
    }
        
    return data;
  }
  boost::multi_array<float, 4> load_vector_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, bool normalize) override
  {
    std::cout << "Vector datasets are unsupported." << std::endl;
    return boost::multi_array<float, 4>();
  }
  boost::multi_array<float, 4> load_tensor_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, bool normalize) override
  {
    boost::multi_array<float, 4> data(boost::extents[size[0]][size[1]][size[2]][1]);

    if (!file_.isValid() || dataset_path.empty())
      return data;

    dataset_path = boost::replace_first_copy(dataset_path, "%Slice%", "%04d");

    for (auto z = 0; z < size[2]; z++)
    {
      auto dataset = file_.getDataSet((boost::format(dataset_path) % (z + offset[2])).str());
      auto count   = dataset.getSpace().getDimensions()[2];
      
      if (z == 0)
        data.resize(boost::extents[size[0]][size[1]][size[2]][count]);
      
      boost::multi_array<float, 3> slice_data;
      dataset
        .select({offset[0], offset[1], 0}, {size[0], size[1], count})
        .read  (slice_data);
      data[boost::indices[index_range()][index_range()][z][index_range()]] = slice_data;
    }

    if (normalize)
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

    dataset_path = boost::replace_first_copy(dataset_path, "%Slice%", "%04d");

    auto shape     = data.shape();
    auto cast_data = const_cast<boost::multi_array<float, 3>&>(data);
    for (auto z = 0; z < shape[2]; z++)
    {
      auto slice_path = (boost::format(dataset_path) % (z + offset[2])).str();
      boost::multi_array<float, 2> slice_data(cast_data[boost::indices[index_range()][index_range()][z]]);
      try
      {
        file_
          .getDataSet   (slice_path)
          .select       ({offset[0], offset[1]}, {shape[0], shape[1]})
          .write        (slice_data);
      }
      catch (...)
      {
        file_
          .createDataSet(slice_path, HighFive::DataSpace({offset[0] + shape[0], offset[1] + shape[1]}), HighFive::AtomicType<float>())
          .select       ({offset[0], offset[1]}, {shape[0], shape[1]})
          .write        (slice_data);
      }
    }
  } 
  void save_vector_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 4>& data) override
  {
    std::cout << "Vector datasets are unsupported." << std::endl;
  }
  void save_tensor_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 4>& data) override 
  {
    if (!file_.isValid() || dataset_path.empty())
      return;

    dataset_path = boost::replace_first_copy(dataset_path, "%Slice%", "%04d");

    auto shape     = data.shape();
    auto cast_data = const_cast<boost::multi_array<float, 4>&>(data);
    for (auto z = 0; z < shape[2]; z++)
    {
      auto slice_path = (boost::format(dataset_path) % (z + offset[2])).str();
      boost::multi_array<float, 3> slice_data(cast_data[boost::indices[index_range()][index_range()][z][index_range()]]);
      try
      {
        file_
          .getDataSet   (slice_path)
          .select       ({offset[0], offset[1], 0}, {shape[0], shape[1], shape[3]})
          .write        (slice_data);
      }
      catch (...)
      {
        file_
          .createDataSet(slice_path, HighFive::DataSpace({offset[0] + shape[0], offset[1] + shape[1], shape[3]}), HighFive::AtomicType<float>())
          .select       ({offset[0], offset[1], 0}, {shape[0], shape[1], shape[3]})
          .write        (slice_data);
      }
    }
  }
};
}

#endif
