#include /* implements */ <ui/plugins/plugin.hpp>

namespace pli
{
plugin::plugin(QWidget* parent) : QWidget(parent)
{

}

void plugin::set_owner(pli::window* owner)
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
