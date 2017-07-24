#ifndef PLI_VIS_IO_SLICE_IMPL_HPP_
#define PLI_VIS_IO_SLICE_IMPL_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>

#include <boost/algorithm/string/replace.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>

namespace pli
{
// Implementation for Vervet1818 type data (where each slice is a separate dataset).
class io_slice_impl
{
public:
  typedef boost::multi_array_types::index_range index_range;

  static std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_scalar_dataset_bounds(
    const HighFive::File& file        , 
    const std::string&    dataset_path)
  {
    if (!file.isValid() || dataset_path.empty())
      return {{0, 0, 0}, {0, 0, 0}};
    auto slice_offset     = boost::lexical_cast<std::size_t>(file.listObjectNames()[0]);
    auto slice_count      = file.getNumberObjects();
    auto slice_dimensions = file.getDataSet((boost::format(boost::replace_first_copy(dataset_path, "%Slice%", "%04d")) % slice_offset).str()).getSpace().getDimensions();
    return {{0, 0, slice_offset}, {slice_dimensions[0], slice_dimensions[1], slice_offset + slice_count}};
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
    auto slice_offset     = boost::lexical_cast<std::size_t>(file.listObjectNames()[0]);
    auto slice_count      = file.getNumberObjects();
    auto slice_dimensions = file.getDataSet((boost::format(boost::replace_first_copy(dataset_path, "%Slice%", "%04d")) % slice_offset).str()).getSpace().getDimensions();
    return {{0, 0, slice_offset, 0}, {slice_dimensions[0], slice_dimensions[1], slice_offset + slice_count, slice_dimensions[2]}};
  }

  static boost::multi_array<float, 3> load_scalar_dataset(
    const HighFive::File&             file        , 
    const std::string&                dataset_path, 
    const std::array<std::size_t, 3>& offset      , 
    const std::array<std::size_t, 3>& size        , 
    const std::array<std::size_t, 3>& stride      , 
    bool                              normalize   )
  {
    boost::multi_array<float, 3> data(boost::extents[size[0]][size[1]][size[2]]);

    if (!file.isValid() || dataset_path.empty())
      return data;

    for (std::size_t z = 0; z < size[2]; z+= stride[2])
    {
      boost::multi_array<float, 2> slice_data;
      file
        .getDataSet((boost::format(boost::replace_first_copy(dataset_path, "%Slice%", "%04d")) % (z + offset[2])).str())
        .select    ({offset[0], offset[1]}, {size[0], size[1]}, std::vector<std::size_t>{stride[0], stride[1]})
        .read      (slice_data);
      data[boost::indices[index_range()][index_range()][z]] = slice_data;
    }

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
    boost::multi_array<float, 4> data(boost::extents[size[0]][size[1]][size[2]][1]);

    if (!file.isValid() || dataset_path.empty())
      return data;

    for (std::size_t z = 0; z < size[2]; z+= stride[2])
    {
      boost::multi_array<float, 3> slice_data;
      auto dataset = file.getDataSet((boost::format(boost::replace_first_copy(dataset_path, "%Slice%", "%04d")) % (z + offset[2])).str());
      auto count   = dataset.getSpace().getDimensions()[2];
      
      if (z == 0) 
        data.resize(boost::extents[size[0]][size[1]][size[2]][count]);   
      
      dataset
        .select({offset[0], offset[1], 0}, {size[0], size[1], count}, std::vector<std::size_t>{stride[0], stride[1], 1})
        .read  (slice_data);
      data[boost::indices[index_range()][index_range()][z][index_range()]] = slice_data;
    }

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

    auto shape     = data.shape();
    auto cast_data = const_cast<boost::multi_array<float, 3>&>(data);
    for (auto z = 0; z < shape[2]; z++)
    {
      auto slice_path = (boost::format(boost::replace_first_copy(dataset_path, "%Slice%", "%04d")) % (z + offset[2])).str();
      boost::multi_array<float, 2> slice_data(cast_data[boost::indices[index_range()][index_range()][z]]);
      try
      {
        file
          .getDataSet   (slice_path)
          .select       ({offset[0], offset[1]}, {shape[0], shape[1]})
          .write        (slice_data);
      }
      catch (...)
      {
        file
          .createDataSet(slice_path, HighFive::DataSpace({offset[0] + shape[0], offset[1] + shape[1]}), HighFive::AtomicType<float>())
          .select       ({offset[0], offset[1]}, {shape[0], shape[1]})
          .write        (slice_data);
      }
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

    auto shape     = data.shape();
    auto cast_data = const_cast<boost::multi_array<float, 4>&>(data);
    for (auto z = 0; z < shape[2]; z++)
    {
      auto slice_path = (boost::format(boost::replace_first_copy(dataset_path, "%Slice%", "%04d")) % (z + offset[2])).str();
      boost::multi_array<float, 3> slice_data(cast_data[boost::indices[index_range()][index_range()][z][index_range()]]);
      try
      {
        file
          .getDataSet   (slice_path)
          .select       ({offset[0], offset[1], 0}, {shape[0], shape[1], shape[3]})
          .write        (slice_data);
      }
      catch (...)
      {
        file
          .createDataSet(slice_path, HighFive::DataSpace({offset[0] + shape[0], offset[1] + shape[1], shape[3]}), HighFive::AtomicType<float>())
          .select       ({offset[0], offset[1], 0}, {shape[0], shape[1], shape[3]})
          .write        (slice_data);
      }
    }
  }
};
}

#endif
