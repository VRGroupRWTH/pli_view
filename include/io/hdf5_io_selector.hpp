#ifndef PLI_IO_HDF5_IO_SELECTOR_HPP_
#define PLI_IO_HDF5_IO_SELECTOR_HPP_

#include <string>
#include <memory>

#include "hdf5_io.hpp"
#include "hdf5_io_2.hpp"
#include "hdf5_io_base.hpp"

namespace pli
{
class hdf5_io_selector
{
public:
  static std::unique_ptr<hdf5_io_base> select(const std::string& filepath)
  {
    if (filepath.substr(0, 3) == "MSA")
      return std::make_unique<hdf5_io_2>(filepath);
    else
      return std::make_unique<hdf5_io>  (filepath);
  }
};
}

#endif
