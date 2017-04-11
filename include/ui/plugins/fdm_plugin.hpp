#ifndef PLI_VIS_FDM_PLUGIN_HPP_
#define PLI_VIS_FDM_PLUGIN_HPP_

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_fdm_toolbox.h>

namespace pli
{
class odf_field;

class fdm_plugin : 
  public plugin, 
  public loggable<fdm_plugin>, 
  public Ui_fdm_toolbox
{
public:
  fdm_plugin(QWidget* parent = nullptr);
  void start () override;

private:
  void update       () const;
  void select_depths() const;

  odf_field* odf_field_;
};
}

#endif
