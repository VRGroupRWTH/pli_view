#ifndef PLI_VIS_FOM_PLUGIN_HPP_
#define PLI_VIS_FOM_PLUGIN_HPP_

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_fom_toolbox.h>

namespace pli
{
class vector_field;

class fom_plugin : 
  public plugin, 
  public loggable<fom_plugin>, 
  public Ui_fom_toolbox
{
public:
  fom_plugin(QWidget* parent = nullptr);
  void start () override;

private:
  void update() const;

  vector_field* vector_field_;
};
}

#endif
