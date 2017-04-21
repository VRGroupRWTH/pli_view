#ifndef PLI_VIS_SELECTOR_HPP_
#define PLI_VIS_SELECTOR_HPP_

#include <array>
#include <cstddef>

#include <QWidget>

#include <attributes/loggable.hpp>
#include <ui_selector.h>

namespace pli
{
class window;

class selector : public QWidget, public Ui_selector, public loggable<selector>
{
  Q_OBJECT
public:
  selector(QWidget* parent = nullptr);

  void set_owner(pli::window* window);

signals:
  void on_change(const std::array<std::size_t, 3>& offset, const std::array<std::size_t, 3> size);

protected:
  void trigger();

  pli::window* owner_ = nullptr;
};
}

#endif
