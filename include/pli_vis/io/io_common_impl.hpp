#ifndef PLI_VIS_IO_BASE_IMPL_HPP_
#define PLI_VIS_IO_BASE_IMPL_HPP_

#include <array>
#include <string>

namespace pli
{
class io_common_impl
{
public:
  template<typename attribute_type>
  static attribute_type load_attribute(
    const HighFive::File& file          , 
    const std::string&    attribute_path)
  {
    attribute_type attribute;
    if (file.isValid() && !attribute_path.empty())
      file.getAttribute(attribute_path).read(attribute);
    return attribute;
  }

  template<typename attribute_type, std::size_t size>
  static void           save_attribute(
    HighFive::File&                         file          ,
    const std::string&                      attribute_path, 
    const std::array<attribute_type, size>& attribute     )
  {
    if (!file.isValid() || attribute_path.empty()) 
      return;
    auto cast_attribute = const_cast<std::array<attribute_type, size>&>(attribute);
    try
    {
      file.getAttribute(attribute_path).write(cast_attribute);
    }
    catch (...)
    {
      file.createAttribute(attribute_path, HighFive::DataSpace(size), HighFive::AtomicType<attribute_type>()).write(cast_attribute);
    }
  }
};
}

#endif
