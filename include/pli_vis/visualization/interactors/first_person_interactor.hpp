#ifndef PLI_VIS_FIRST_PERSON_INTERACTOR_HPP_
#define PLI_VIS_FIRST_PERSON_INTERACTOR_HPP_

#include <map>

#include <QPoint>

#include <pli_vis/visualization/interactors/interactor.hpp>

class QKeyEvent;
class QMouseEvent;

namespace pli
{
class first_person_interactor : public interactor
{
public:
  first_person_interactor(camera* camera);

  void update_transform   ()                   override;
  void key_press_handler  (QKeyEvent*   event) override;
  void key_release_handler(QKeyEvent*   event) override;
  void mouse_press_handler(QMouseEvent* event) override;
  void mouse_move_handler (QMouseEvent* event) override;

private:
  std::map<int, bool> key_map_;

  QPoint last_mouse_position_;
};
}

#endif