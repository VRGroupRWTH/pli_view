#ifndef PLI_VIS_ZERNIKE_PLUGIN_HPP_
#define PLI_VIS_ZERNIKE_PLUGIN_HPP_

#include <future>

#include <vector_types.h>

#include <pli_vis/ui/plugin.hpp>
#include <ui_zernike_toolbox.h>

namespace pli
{
class tensor_field;

class zernike_plugin : public plugin<zernike_plugin, Ui_zernike_toolbox>
{
public:
  struct parameters
  {
    uint2    vectors_size   ;
    uint2    superpixel_size;
    uint2    partitions     ;
    unsigned maximum_degree ;
  };

  explicit zernike_plugin(QWidget* parent = nullptr);

  void       start         () override;
  parameters get_parameters() const   ;

private:
  std::future<void> future_       ;
  tensor_field*     tensor_field_ = nullptr;
};
}

#endif
