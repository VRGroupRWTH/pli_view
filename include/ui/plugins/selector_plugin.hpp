#ifndef PLI_VIS_SELECTOR_HPP_
#define PLI_VIS_SELECTOR_HPP_

#include <array>
#include <cstddef>
#include <memory>

#include <attributes/loggable.hpp>
#include <ui/selection_square.hpp>
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
  
  std::array<std::size_t, 3> selection_offset() const;
  std::array<std::size_t, 3> selection_size  () const;

  void start() override;

signals:
  void on_change(const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3>& size);

private:
  void upload();

  std::unique_ptr<selection_square> selection_square_;
};
}

#endif
