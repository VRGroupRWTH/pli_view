#ifndef PLI_VIS_WINDOW_HPP_
#define PLI_VIS_WINDOW_HPP_

#include <memory>

#include <QMainWindow>

#include <hdf5/hdf5_io.hpp>

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
  void bind_actions (); 
  void update_viewer();

  Ui::window ui_;
  QWidget    main_;

  std::unique_ptr<pli::hdf5_io<float>> io_;
  std::array<std::size_t, 3>           offset_;
  std::array<std::size_t, 3>           size_;
};
}

#endif
