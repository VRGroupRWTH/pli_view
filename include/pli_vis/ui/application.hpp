#ifndef PLI_VIS_WINDOW_HPP_
#define PLI_VIS_WINDOW_HPP_

#include <vector>

#include <QMainWindow>

#include <pli_vis/aspects/loggable.hpp>
#include <pli_vis/ui/plugin.hpp>

#include <ui_window.h>

namespace pli
{
class plugin;

class application : public QMainWindow, public Ui_window, public loggable<application>
{
public:
   application();
  ~application();

  template<typename plugin_type>
  plugin_type* get_plugin()
  {
    for (auto plugin : plugins_)
      if (typeid(*plugin) == typeid(plugin_type))
        return reinterpret_cast<plugin_type*>(plugin);
    return nullptr;
  }

private:
  void bind_actions();

  std::vector<plugin*> plugins_;
};
}

#endif
