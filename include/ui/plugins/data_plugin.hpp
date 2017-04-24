#ifndef PLI_VIS_DATA_PLUGIN_HPP_
#define PLI_VIS_DATA_PLUGIN_HPP_

#include <memory>

#include <attributes/loggable.hpp>
#include <io/hdf5_io_base.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_data_toolbox.h>

namespace pli
{
class data_plugin : public plugin, public loggable<data_plugin>, public Ui_data_toolbox
{
  Q_OBJECT

public:
  data_plugin(QWidget* parent = nullptr);
  void start() override;

  hdf5_io_base* io() const;

signals:
  void on_change(hdf5_io_base* io);

private:
  void set_file(const std::string& filename);
  std::unique_ptr<hdf5_io_base> io_;
};
}

#endif
