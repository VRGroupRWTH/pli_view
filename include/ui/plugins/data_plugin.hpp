#ifndef PLI_VIS_DATA_PLUGIN_HPP_
#define PLI_VIS_DATA_PLUGIN_HPP_

#include <memory>

#include <hdf5/hdf5_io.hpp>

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_data_toolbox.h>

namespace pli
{
class data_plugin : public plugin, public loggable<data_plugin>, public Ui_data_toolbox
{
  Q_OBJECT

public:
  data_plugin(QWidget* parent = nullptr);

  hdf5_io<float>* io() const;

signals:
  void on_change(pli::hdf5_io<float>* io);

private:
  void start() override;

  void set_file(const std::string& filename);

  std::unique_ptr<pli::hdf5_io<float>> io_;
};
}

#endif
