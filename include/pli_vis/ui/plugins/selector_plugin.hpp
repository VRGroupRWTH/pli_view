#ifndef PLI_VIS_SELECTOR_HPP_
#define PLI_VIS_SELECTOR_HPP_

#include <array>
#include <cstddef>

#include <pli_vis/ui/plugin.hpp>
#include <ui_selector_toolbox.h>

namespace pli
{
class application;

class selector_plugin : public plugin<selector_plugin, Ui_selector_toolbox>
{
  Q_OBJECT

public:
  explicit selector_plugin(QWidget* parent = nullptr);
  
  std::array<std::size_t, 3> selection_offset() const;
  std::array<std::size_t, 3> selection_size  () const;
  std::array<std::size_t, 3> selection_stride() const;

  void start() override;

signals:
  void on_change(
    const std::array<std::size_t, 3>& offset, 
    const std::array<std::size_t, 3>& size  , 
    const std::array<std::size_t, 3>& stride);
};
}

#endif
