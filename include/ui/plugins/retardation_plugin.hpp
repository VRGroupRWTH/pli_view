#ifndef PLI_VIS_RETARDATION_PLUGIN_HPP_
#define PLI_VIS_RETARDATION_PLUGIN_HPP_

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_retardation_toolbox.h>

namespace pli
{
class scalar_field;

class retardation_plugin : 
  public plugin, 
  public loggable<retardation_plugin>,
  public Ui_retardation_toolbox
{
public:
  retardation_plugin(QWidget* parent = nullptr);
  void start        () override;

private:
  void update       () const;

  scalar_field* scalar_field_;
};
}

#endif
