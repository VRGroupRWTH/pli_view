#ifndef PLI_VIS_SELECTOR_HPP_
#define PLI_VIS_SELECTOR_HPP_

#include <array>
#include <cstddef>

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_selector_toolbox.h>

namespace pli
{
class window;

class selector_plugin : public plugin, public Ui_selector_toolbox, public loggable<selector_plugin>
{
  Q_OBJECT
public:
  selector_plugin(QWidget* parent = nullptr);
  void start() override;

  std::array<std::size_t, 3> offset() const;
  std::array<std::size_t, 3> size  () const;

signals:
  void on_change(const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size);
};
}

#endif
