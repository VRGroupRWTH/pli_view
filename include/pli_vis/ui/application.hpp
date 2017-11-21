#ifndef PLI_VIS_WINDOW_HPP_
#define PLI_VIS_WINDOW_HPP_

#include <vector>

#include <QLabel>
#include <QMainWindow>
#include <QProgressBar>

#include <pli_vis/aspects/loggable.hpp>
#include <pli_vis/ui/widgets/wait_spinner.hpp>
#include <pli_vis/ui/plugin_base.hpp>
#include <ui_window.h>

namespace pli
{
class application : public QMainWindow, public Ui_window, public loggable<application>
{
public:
   application();
  ~application();

  void set_is_loading(const bool is_loading) const;

  template<typename plugin_type>
  plugin_type* get_plugin()
  {
    for (auto plugin : plugins_)
      if (typeid(*plugin) == typeid(plugin_type))
        return reinterpret_cast<plugin_type*>(plugin);
    return nullptr;
  }

private:
  void bind_actions         ();
  void create_gpu_status_bar();

  std::vector<plugin_base*> plugins_;
  pli::wait_spinner* wait_spinner_;

  QLabel*       gpu_status_label_ = nullptr;
  QProgressBar* gpu_status_bar_   = nullptr;
};
}

#endif
