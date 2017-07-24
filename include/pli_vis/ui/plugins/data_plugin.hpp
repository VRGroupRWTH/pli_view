#ifndef PLI_VIS_DATA_PLUGIN_HPP_
#define PLI_VIS_DATA_PLUGIN_HPP_

#include <future>

#include <pli_vis/io/io.hpp>
#include <pli_vis/ui/plugin.hpp>
#include <ui_data_toolbox.h>

namespace pli
{
class data_plugin : public plugin<data_plugin, Ui_data_toolbox>
{
  Q_OBJECT

public:
  explicit data_plugin(QWidget* parent = nullptr);

  void start() override;

signals:
  void on_change();
  void on_load  ();

private:
  io                io_    ;
  std::future<void> future_;
};
}

#endif
