#ifndef PLI_VIS_WINDOW_HPP_
#define PLI_VIS_WINDOW_HPP_

#include <vector>

#include <QMainWindow>

#include <attributes/loggable.hpp>
#include <ui_window.h>

namespace pli
{
class plugin;

class window : public QMainWindow, public Ui_window, public loggable<window>
{
public:
   window();
  ~window();

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
