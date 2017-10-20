#ifndef PLI_VIS_ZERNIKE_PLUGIN_HPP_
#define PLI_VIS_ZERNIKE_PLUGIN_HPP_

#include <pli_vis/ui/plugin.hpp>
#include <ui_zernike_toolbox.h>

namespace pli
{
class zernike_plugin : public plugin<zernike_plugin, Ui_zernike_toolbox>
{
public:
  explicit zernike_plugin(QWidget* parent = nullptr);
  void start() override;
};
}

#endif
