#ifndef PLI_VIS_SIMPLE_INTERACTOR_HPP_
#define PLI_VIS_SIMPLE_INTERACTOR_HPP_

#include <QPoint>

#include <pli_vis/visualization/interactors/interactor.hpp>

class QKeyEvent;
class QMouseEvent;

namespace pli
{
class simple_interactor : public interactor
{
public:
  simple_interactor(camera* camera);

  void mouse_press_handler(QMouseEvent* event) override;
  void mouse_move_handler (QMouseEvent* event) override;

private:
  QPoint last_mouse_position_;
};
}

#endif