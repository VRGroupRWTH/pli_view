#ifndef PLI_VIS_REMOTE_VIEWER_HPP_
#define PLI_VIS_REMOTE_VIEWER_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <future>
#include <string>

#include <boost/signals2.hpp>
#include <glm/glm.hpp>
#include <QCloseEvent>
#include <QImage>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QTimer>

#include <pli_vis/visualization/interactors/interactor.hpp>

namespace pli
{
class application;

class remote_viewer : public QLabel
{
public:
  explicit remote_viewer  (const std::string& address, application* owner, interactor* interactor, QWidget* parent = nullptr);
  remote_viewer           (const remote_viewer&  that) = default;
  remote_viewer           (      remote_viewer&& temp) = default;
  virtual ~remote_viewer  ();
  remote_viewer& operator=(const remote_viewer&  that) = default;
  remote_viewer& operator=(      remote_viewer&& temp) = default;

  void closeEvent     (QCloseEvent* event) override;
  void keyPressEvent  (QKeyEvent*   event) override;
  void keyReleaseEvent(QKeyEvent*   event) override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent (QMouseEvent* event) override;

  boost::signals2::signal<void()> on_close ;
  boost::signals2::signal<void()> on_render;

protected:
  std::string                address_          ;
  application*               owner_            ;
  interactor*                interactor_       ;
  std::atomic<bool>          alive_            ;
  std::future<void>          future_           ;

  std::string                filepath_         ;
  std::array<std::size_t, 3> offset_           ;
  std::array<std::size_t, 3> size_             ;
  std::array<std::size_t, 3> stride_           ;
  float                      step_             ;
  std::size_t                iterations_       ;
  std::array<std::size_t, 3> seed_offset_      ;
  std::array<std::size_t, 3> seed_size_        ;
  std::array<std::size_t, 3> seed_stride_      ;
  int                        color_mapping_    ;
  float                      k_                ;
  glm::vec3                  translation_      ;
  glm::vec3                  forward_          ;
  glm::vec3                  up_               ;
  std::array<std::size_t, 2> image_size_       ;
  float                      streamline_radius_;
  QImage                     image_            ;
  QTimer                     timer_            ;

};
}

#endif