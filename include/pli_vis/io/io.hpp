#ifndef PLI_VIS_IO_HPP_
#define PLI_VIS_IO_HPP_

#define H5_USE_BOOST_MULTI_ARRAY

#include <array>
#include <string>

#include <boost/multi_array.hpp>

#include <pli_vis/third_party/highfive/H5File.hpp>
#include <pli_vis/io/io_slice_impl.hpp>
#include <pli_vis/io/io_volume_impl.hpp>

namespace pli
{
class io
{
public:
  explicit io(
    std::string filepath            = std::string(),
    std::string vector_spacing_path = std::string(),
    std::string transmittance_path  = std::string(),
    std::string retardation_path    = std::string(),
    std::string direction_path      = std::string(),
    std::string inclination_path    = std::string(),
    std::string mask_path           = std::string(),
    std::string unit_vector_path    = std::string(),
    std::string distribution_path   = std::string())
    : filepath_           (filepath           )
    , vector_spacing_path_(vector_spacing_path)
    , transmittance_path_ (transmittance_path )
    , retardation_path_   (retardation_path   )
    , direction_path_     (direction_path     )
    , inclination_path_   (inclination_path   )
    , mask_path_          (mask_path          )
    , unit_vector_path_   (unit_vector_path   )
    , distribution_path_  (distribution_path  )
  {
    try         { file_ = std::make_unique<HighFive::File>(filepath_, HighFive::File::ReadWrite); }
    catch (...) { file_ = nullptr; std::cout << "Invalid file" << std::endl; }
  }
  virtual ~io() = default;
  
  void set_filepath           (const std::string& filepath           )
  {
    filepath_ = filepath;
    try        { file_ = std::make_unique<HighFive::File>(filepath_, HighFive::File::ReadWrite); }
    catch(...) { file_ = nullptr; std::cout << "Invalid file" << std::endl; }
  }
  void set_vector_spacing_path(const std::string& vector_spacing_path)
  {
    vector_spacing_path_   = vector_spacing_path;
  }
  void set_transmittance_path (const std::string& transmittance_path )
  {
    transmittance_path_      = transmittance_path;
  }
  void set_retardation_path   (const std::string& retardation_path   )
  {
    retardation_path_        = retardation_path;
  }
  void set_direction_path     (const std::string& direction_path     )
  {
    direction_path_    = direction_path;
  }
  void set_inclination_path   (const std::string& inclination_path   )
  {
    inclination_path_  = inclination_path;
  }
  void set_mask_path          (const std::string& mask_path          )
  {
    mask_path_               = mask_path;
  }
  void set_unit_vector_path   (const std::string& unit_vector_path   )
  {
    unit_vector_path_ = unit_vector_path;
  }
  void set_distribution_path  (const std::string& distribution_path  )
  {
    distribution_path_ = distribution_path;
  }

  const std::string& filepath           () const
  {
    return filepath_;
  }
  const std::string& vector_spacing_path() const
  {
    return vector_spacing_path_;
  }
  const std::string& transmittance_path () const
  {
    return transmittance_path_;
  }
  const std::string& retardation_path   () const
  {
    return retardation_path_;
  }
  const std::string& direction_path     () const
  {
    return direction_path_;
  }
  const std::string& inclination_path   () const
  {
    return inclination_path_;
  }
  const std::string& mask_path          () const
  {
    return mask_path_;
  }
  const std::string& unit_vector_path   () const
  {
    return unit_vector_path_;
  }
  const std::string& distribution_path  () const
  {
    return distribution_path_;
  }

  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_transmittance_bounds() const
  {
    if(!file_) return std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>();
    try        {       return io_slice_impl ::load_scalar_dataset_bounds(*file_, transmittance_path_); }
    catch(...) { try { return io_volume_impl::load_scalar_dataset_bounds(*file_, transmittance_path_); } catch(...) { return std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>(); } }
  }
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_retardation_bounds  () const
  {
    if (!file_) return std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>();
    try        {       return io_slice_impl ::load_scalar_dataset_bounds(*file_, retardation_path_); }
    catch(...) { try { return io_volume_impl::load_scalar_dataset_bounds(*file_, retardation_path_); } catch(...) { return std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>(); } }
  }
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_direction_bounds    () const
  {
    if (!file_) return std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>();
    try        {       return io_slice_impl ::load_scalar_dataset_bounds(*file_, direction_path_); }
    catch(...) { try { return io_volume_impl::load_scalar_dataset_bounds(*file_, direction_path_); } catch(...) { return std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>(); } }
  }
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_inclination_bounds  () const
  {
    if (!file_) return std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>();
    try        {       return io_slice_impl ::load_scalar_dataset_bounds(*file_, inclination_path_); }
    catch(...) { try { return io_volume_impl::load_scalar_dataset_bounds(*file_, inclination_path_); } catch(...) { return std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>(); } }
  }
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> load_mask_bounds         () const
  {
    if (!file_) return std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>();
    try        {       return io_slice_impl ::load_scalar_dataset_bounds(*file_, mask_path_); }
    catch(...) { try { return io_volume_impl::load_scalar_dataset_bounds(*file_, mask_path_); } catch(...) { return std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>(); } }
  }
  std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> load_unit_vector_bounds  () const
  {
    if (!file_) return std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>>();
    try        {       return io_slice_impl ::load_vector_dataset_bounds(*file_, unit_vector_path_); }
    catch(...) { try { return io_volume_impl::load_vector_dataset_bounds(*file_, unit_vector_path_); } catch(...) { return std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>>(); } }
  }
  std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> load_distribution_bounds () const
  {
    if (!file_) return std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>>();
    try        {       return io_slice_impl ::load_tensor_dataset_bounds(*file_, distribution_path_); }
    catch(...) { try { return io_volume_impl::load_tensor_dataset_bounds(*file_, distribution_path_); } catch(...) { return std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>>(); } }
  }

  boost::multi_array<float, 3> load_transmittance(const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true) const
  {
    if (!file_) return boost::multi_array<float, 3>();
    try        {       return io_slice_impl ::load_scalar_dataset(*file_, transmittance_path_, offset, size, stride, normalize); }
    catch(...) { try { return io_volume_impl::load_scalar_dataset(*file_, transmittance_path_, offset, size, stride, normalize); } catch(...) { return boost::multi_array<float, 3>(); } }
  } 
  boost::multi_array<float, 3> load_retardation  (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true) const
  {
    if (!file_) return boost::multi_array<float, 3>();
    try        {       return io_slice_impl ::load_scalar_dataset(*file_, retardation_path_  , offset, size, stride, normalize); }
    catch(...) { try { return io_volume_impl::load_scalar_dataset(*file_, retardation_path_  , offset, size, stride, normalize); } catch(...) { return boost::multi_array<float, 3>(); } }
  }
  boost::multi_array<float, 3> load_direction    (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true) const
  {
    if (!file_) return boost::multi_array<float, 3>();
    try        {       return io_slice_impl ::load_scalar_dataset(*file_, direction_path_, offset, size, stride, normalize); }
    catch(...) { try { return io_volume_impl::load_scalar_dataset(*file_, direction_path_, offset, size, stride, normalize); } catch(...) { return boost::multi_array<float, 3>(); } }
  }
  boost::multi_array<float, 3> load_inclination  (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true) const
  {
    if (!file_) return boost::multi_array<float, 3>();
    try        {       return io_slice_impl ::load_scalar_dataset(*file_, inclination_path_, offset, size, stride, normalize); }
    catch(...) { try { return io_volume_impl::load_scalar_dataset(*file_, inclination_path_, offset, size, stride, normalize); } catch(...) { return boost::multi_array<float, 3>(); } }
  } 
  boost::multi_array<float, 3> load_mask         (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true) const
  {
    if (!file_) return boost::multi_array<float, 3>();
    try        {       return io_slice_impl ::load_scalar_dataset(*file_, mask_path_, offset, size, stride, normalize); }
    catch(...) { try { return io_volume_impl::load_scalar_dataset(*file_, mask_path_, offset, size, stride, normalize); } catch(...) { return boost::multi_array<float, 3>(); } }
  }
  boost::multi_array<float, 4> load_unit_vector  (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true) const
  {
    if (!file_) return boost::multi_array<float, 4>();
    try        {       return io_slice_impl ::load_vector_dataset(*file_, unit_vector_path_, offset, size, stride, normalize); }
    catch(...) { try { return io_volume_impl::load_vector_dataset(*file_, unit_vector_path_, offset, size, stride, normalize); } catch(...) { return boost::multi_array<float, 4>(); } }
  }
  boost::multi_array<float, 4> load_distribution (const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size, const std::array<std::size_t, 3>& stride = {1,1,1}, bool normalize = true) const
  {
    if (!file_) return boost::multi_array<float, 4>();
    try        {       return io_slice_impl ::load_tensor_dataset(*file_, distribution_path_, offset, size, stride, normalize); }
    catch(...) { try { return io_volume_impl::load_tensor_dataset(*file_, distribution_path_, offset, size, stride, normalize); } catch(...) { return boost::multi_array<float, 4>(); } }
  }
 
  void save_transmittance(const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data)
  {
    if (!file_) return;
    try        {       io_slice_impl ::save_scalar_dataset(*file_, transmittance_path_, offset, data); }
    catch(...) { try { io_volume_impl::save_scalar_dataset(*file_, transmittance_path_, offset, data); } catch(...) { } }
  }
  void save_retardation  (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data)
  {
    if (!file_) return;
    try        {       io_slice_impl ::save_scalar_dataset(*file_, retardation_path_, offset, data); }
    catch(...) { try { io_volume_impl::save_scalar_dataset(*file_, retardation_path_, offset, data); } catch(...) { } }
  }
  void save_direction    (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data)
  {
    if (!file_) return;
    try        {       io_slice_impl ::save_scalar_dataset(*file_, direction_path_, offset, data); }
    catch(...) { try { io_volume_impl::save_scalar_dataset(*file_, direction_path_, offset, data); } catch(...) { } }
  }
  void save_inclination  (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data)
  {
    if (!file_) return;
    try        {       io_slice_impl ::save_scalar_dataset(*file_, inclination_path_, offset, data); }
    catch(...) { try { io_volume_impl::save_scalar_dataset(*file_, inclination_path_, offset, data); } catch(...) { } }
  }
  void save_mask         (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 3>& data)
  {
    if (!file_) return;
    try        {       io_slice_impl ::save_scalar_dataset(*file_, mask_path_, offset, data); }
    catch(...) { try { io_volume_impl::save_scalar_dataset(*file_, mask_path_, offset, data); } catch(...) { } }
  }  
  void save_unit_vector  (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 4>& data)
  {
    if (!file_) return;
    try        {       io_slice_impl ::save_vector_dataset(*file_, unit_vector_path_, offset, data); }
    catch(...) { try { io_volume_impl::save_vector_dataset(*file_, unit_vector_path_, offset, data); } catch(...) { } }
  }
  void save_distribution (const std::array<std::size_t, 3>& offset, const boost::multi_array<float, 4>& data)
  {
    if (!file_) return;
    try        {       io_slice_impl ::save_tensor_dataset(*file_, distribution_path_, offset, data); }
    catch(...) { try { io_volume_impl::save_tensor_dataset(*file_, distribution_path_, offset, data); } catch(...) { } }
  }
  
protected:
  std::string filepath_;
  std::string vector_spacing_path_;
  std::string transmittance_path_;
  std::string retardation_path_;
  std::string direction_path_;
  std::string inclination_path_;
  std::string mask_path_;
  std::string unit_vector_path_;
  std::string distribution_path_;
  std::unique_ptr<HighFive::File> file_;
};
}

#endif
