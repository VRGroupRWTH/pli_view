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
#include <QLabel>

namespace pli
{
class application;

class remote_viewer : public QLabel
{
public:
  explicit remote_viewer  (application* owner, QWidget* parent = nullptr);
  remote_viewer           (const remote_viewer&  that) = default;
  remote_viewer           (      remote_viewer&& temp) = default;
  virtual ~remote_viewer  ();
  remote_viewer& operator=(const remote_viewer&  that) = default;
  remote_viewer& operator=(      remote_viewer&& temp) = default;

  void closeEvent(QCloseEvent* event) override;

  boost::signals2::signal<void()> on_close;

protected:
  std::string       address_ = "tcp://localhost:5555";
  application*      owner_   ;
  std::atomic<bool> alive_   ;
  std::future<void> future_  ;

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

};
}

#endif