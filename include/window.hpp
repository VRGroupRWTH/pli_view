#ifndef PLI_VIS_WINDOW_HPP_
#define PLI_VIS_WINDOW_HPP_

#include <memory>

#include <QMainWindow>

#include <ui_window.h>

#include <attributes/loggable.hpp>

namespace pli
{
class window : public QMainWindow, public loggable<window>
{
public:
   window();
  ~window();

  const Ui::window& ui() const
  {
    return ui_;
  }

private:
  void bind_actions();

  Ui::window ui_;
  QWidget    main_;
};
}

#endif
