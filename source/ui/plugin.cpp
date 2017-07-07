#include <pli_vis/ui/plugin.hpp>

namespace pli
{
plugin::plugin(QWidget* parent) : QWidget(parent)
{

}

void plugin::set_owner(pli::application* owner)
{
  owner_ = owner;
}

void plugin::awake  ()
{

}
void plugin::start  ()
{

}
void plugin::destroy()
{

}
}
