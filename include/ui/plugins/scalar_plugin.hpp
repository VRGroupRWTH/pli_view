#ifndef PLI_VIS_SCALAR_PLUGIN_HPP_
#define PLI_VIS_SCALAR_PLUGIN_HPP_

#include <map>

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_scalar_toolbox.h>

namespace pli
{
class scalar_field;

class scalar_plugin : 
  public plugin, 
  public loggable<scalar_plugin>,
  public Ui_scalar_toolbox
{
public:
  scalar_plugin(QWidget* parent = nullptr);
  void start        () override;

private:
  void update       () const;

  std::map<std::string, scalar_field*> scalar_fields_;
};
}

#endif
