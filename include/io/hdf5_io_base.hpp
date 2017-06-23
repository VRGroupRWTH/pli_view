#ifndef PLI_IO_HDF5_IO_COMMON_HPP_
#define PLI_IO_HDF5_IO_COMMON_HPP_

#define H5_USE_BOOST_MULTI_ARRAY

#include <array>
#include <string>

#include <boost/multi_array.hpp>

#include <third_party/highfive/H5Attribute.hpp>
#include <third_party/highfive/H5File.hpp>

namespace pli
{
class hdf5_io_base
{
public:
  explicit hdf5_io_base(
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
    : filepath_                        (filepath                       )
    , attribute_path_vector_spacing_   (attribute_path_vector_spacing  )
    , attribute_path_block_size_       (attribute_path_block_size      )
    , dataset_path_mask_               (dataset_path_mask              )
    , dataset_path_transmittance_      (dataset_path_transmittance     )
    , dataset_path_retardation_        (dataset_path_retardation       )
    , dataset_path_fiber_direction_    (dataset_path_fiber_direction   )
    , dataset_path_fiber_inclination_  (dataset_path_fiber_inclination )
    , dataset_path_fiber_unit_vectors_ (dataset_path_fiber_unit_vectors)
    , dataset_path_fiber_distribution_ (dataset_path_fiber_distribution)
    , file_                            (filepath, HighFive::File::ReadWrite)
  {

  }
  virtual ~hdf5_io_base() = default;
  
  void set_filepath                       (const std::string& filepath                       )
  {
    filepath_ = filepath;
    file_     = HighFive::File(filepath_, HighFive::File::ReadWrite);
  }
  void set_attribute_path_vector_spacing  (const std::string& attribute_path_vector_spacing  )
  {
    attribute_path_vector_spacing_   = attribute_path_vector_spacing;
  }
  void set_attribute_path_block_size      (const std::string& attribute_path_block_size      )
  {
    attribute_path_block_size_ = attribute_path_block_size;
  }
  void set_dataset_path_mask              (const std::string& dataset_path_mask              )
  {
    dataset_path_mask_               = dataset_path_mask;
  }
  void set_dataset_path_transmittance     (const std::string& dataset_path_transmittance     )
  {
    dataset_path_transmittance_      = dataset_path_transmittance;
  }
  void set_dataset_path_retardation       (const std::string& dataset_path_retardation       )
  {
    dataset_path_retardation_        = dataset_path_retardation;
  }
  void set_dataset_path_fiber_direction   (const std::string& dataset_path_fiber_direction   )
  {
    dataset_path_fiber_direction_    = dataset_path_fiber_direction;
  }
  void set_dataset_path_fiber_inclination (const std::string& dataset_path_fiber_inclination )
  {
    dataset_path_fiber_inclination_  = dataset_path_fiber_inclination;
  }
  void set_dataset_path_fiber_unit_vectors(const std::string& dataset_path_fiber_unit_vectors)
  {
    dataset_path_fiber_unit_vectors_ = dataset_path_fiber_unit_vectors;
  }
  void set_dataset_path_fiber_distribution(const std::string& dataset_path_fiber_distribution)
  {
    dataset_path_fiber_distribution_ = dataset_path_fiber_distribution;
  }

  const std::string& filepath                       () const
  {
    return filepath_;
  }
  const std::string& attribute_path_vector_spacing  () const
  {
    return attribute_path_vector_spacing_;
  }
  const std::string& attribute_path_block_size      () const
  {
    return attribute_path_block_size_;
  }
  const std::string& dataset_path_mask              () const
  {
    return dataset_path_mask_;
  }
  const std::string& dataset_path_transmittance     () const
  {
    return dataset_path_transmittance_;
  }
  const std::string& dataset_path_retardation       () const
  {
    return dataset_path_retardation_;
  }
  const std::string& dataset_path_fiber_direction   () const
  {
    return dataset_path_fiber_direction_;
  }
  const std::string& dataset_path_fiber_inclination () const
  {
    return dataset_path_fiber_inclination_;
  }
  const std::string& dataset_path_fiber_unit_vectors() const
  {
    return dataset_path_fiber_unit_vectors_;
  }
  const std::string& dataset_path_fiber_distribution() const
  {
    return dataset_path_fiber_distribution_;
  }

  std::array<float      , 3> load_vector_spacing()
  {
    return load_attribute<std::array<float, 3>>(attribute_path_vector_spacing_);
  }
  std::array<std::size_t, 3> load_block_size    ()
  {
    return load_attribute<std::array<std::size_t, 3>>(attribute_path_block_size_);
  }

  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_mask_dataset_bounds              ()
  {
    return load_scalar_dataset_bounds(dataset_path_mask_);
  }
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_transmittance_dataset_bounds     ()
  {
    return load_scalar_dataset_bounds(dataset_path_transmittance_);
  }
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_retardation_dataset_bounds       ()
  {
    return load_scalar_dataset_bounds(dataset_path_retardation_);
  }
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_fiber_direction_dataset_bounds   ()
  {
    return load_scalar_dataset_bounds(dataset_path_fiber_direction_);
  }
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_fiber_inclination_dataset_bounds ()
  {
    return load_scalar_dataset_bounds(dataset_path_fiber_inclination_);
  }
  std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> load_fiber_unit_vectors_dataset_bounds()
  {
    return load_vector_dataset_bounds(dataset_path_fiber_unit_vectors_);
  }
  std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> load_fiber_distribution_dataset_bounds()
  {
    return load_tensor_dataset_bounds(dataset_path_fiber_distribution_);
  }

  boost::multi_array<float, 3> load_mask_dataset              (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true)
  {
    return load_scalar_dataset(dataset_path_mask_              , offset, size, stride, normalize);
  }
  boost::multi_array<float, 3> load_transmittance_dataset     (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true)
  {
    return load_scalar_dataset(dataset_path_transmittance_     , offset, size, stride, normalize);
  } 
  boost::multi_array<float, 3> load_retardation_dataset       (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true)
  {
    return load_scalar_dataset(dataset_path_retardation_       , offset, size, stride, normalize);
  }
  boost::multi_array<float, 3> load_fiber_direction_dataset   (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true)
  {
    return load_scalar_dataset(dataset_path_fiber_direction_   , offset, size, stride, normalize);
  }
  boost::multi_array<float, 3> load_fiber_inclination_dataset (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true)
  {
    return load_scalar_dataset(dataset_path_fiber_inclination_ , offset, size, stride, normalize);
  } 
  boost::multi_array<float, 4> load_fiber_unit_vectors_dataset(const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true)
  {
    return load_vector_dataset(dataset_path_fiber_unit_vectors_, offset, size, stride, normalize);
  }
  boost::multi_array<float, 4> load_fiber_distribution_dataset(const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true)
  {
    return load_tensor_dataset(dataset_path_fiber_distribution_, offset, size, stride, normalize);
  }
 
  void save_vector_spacing(const std::array<float      , 3>& data)
  {
    return save_attribute(attribute_path_vector_spacing_, data);
  }    
  void save_block_size    (const std::array<std::size_t, 3>& data)
  {
    return save_attribute(attribute_path_block_size_    , data);
  }  
  
  void save_mask_dataset              (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data)
  {
    save_scalar_dataset(dataset_path_mask_              , offset, data);
  }  
  void save_transmittance_dataset     (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data)
  {
    save_scalar_dataset(dataset_path_transmittance_     , offset, data);
  }
  void save_retardation_dataset       (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data)
  {
    save_scalar_dataset(dataset_path_retardation_       , offset, data);
  }
  void save_fiber_direction_dataset   (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data)
  {
    save_scalar_dataset(dataset_path_fiber_direction_   , offset, data);
  }
  void save_fiber_inclination_dataset (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data)
  {
    save_scalar_dataset(dataset_path_fiber_inclination_ , offset, data);
  }
  void save_fiber_unit_vectors_dataset(const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 4>& data)
  {
    save_vector_dataset(dataset_path_fiber_unit_vectors_, offset, data);
  }
  void save_fiber_distribution_dataset(const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 4>& data)
  {
    save_tensor_dataset(dataset_path_fiber_distribution_, offset, data);
  }
   
protected:
  typedef boost::multi_array_types::index_range index_range;

  virtual std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_scalar_dataset_bounds(std::string dataset_path) = 0;
  virtual std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> load_vector_dataset_bounds(std::string dataset_path) = 0;
  virtual std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> load_tensor_dataset_bounds(std::string dataset_path) = 0;

  virtual boost::multi_array<float, 3> load_scalar_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1, 1, 1}, bool normalize = true) = 0;
  virtual boost::multi_array<float, 4> load_vector_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1, 1, 1}, bool normalize = true) = 0;
  virtual boost::multi_array<float, 4> load_tensor_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1, 1, 1}, bool normalize = true) = 0;

  virtual void save_scalar_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data) = 0;
  virtual void save_vector_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 4>& data) = 0;
  virtual void save_tensor_dataset(std::string dataset_path, const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 4>& data) = 0;

  template<typename attribute_type>
  attribute_type load_attribute(std::string attribute_path)
  {
    attribute_type attribute;
    if (file_.isValid() && !attribute_path.empty())
      file_.getAttribute(attribute_path).read(attribute);
    return attribute;
  }

  template<typename attribute_type, std::size_t size>
  void save_attribute(std::string attribute_path, const std::array<attribute_type, size>& attribute)
  {
    if (!file_.isValid() || attribute_path.empty())
      return;

    auto cast_attribute = const_cast<std::array<attribute_type, size>&>(attribute);

    try
    {
      file_
        .getAttribute   (attribute_path)
        .write          (cast_attribute);
    }
    catch (...)
    {
      file_
        .createAttribute(attribute_path, HighFive::DataSpace(size), HighFive::AtomicType<attribute_type>())
        .write          (cast_attribute);
    }
  }

  std::string filepath_;
  std::string attribute_path_vector_spacing_;
  std::string attribute_path_block_size_;
  std::string dataset_path_mask_;
  std::string dataset_path_transmittance_;
  std::string dataset_path_retardation_;
  std::string dataset_path_fiber_direction_;
  std::string dataset_path_fiber_inclination_;
  std::string dataset_path_fiber_unit_vectors_;
  std::string dataset_path_fiber_distribution_;

  HighFive::File file_;
};
}

#endif
