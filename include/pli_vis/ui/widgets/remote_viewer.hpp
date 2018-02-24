#ifndef PLI_VIS_REMOTE_VIEWER_HPP_
#define PLI_VIS_REMOTE_VIEWER_HPP_

#include <atomic>
#include <future>

#include <boost/signals2.hpp>
#include <QCloseEvent>
#include <QLabel>

namespace pli
{
class remote_viewer : public QLabel
{
public:
  explicit remote_viewer  (QWidget* parent = nullptr);
  remote_viewer           (const remote_viewer&  that) = default;
  remote_viewer           (      remote_viewer&& temp) = default;
  virtual ~remote_viewer  ();
  remote_viewer& operator=(const remote_viewer&  that) = default;
  remote_viewer& operator=(      remote_viewer&& temp) = default;

  void closeEvent(QCloseEvent* event) override;

  boost::signals2::signal<void()> on_close;

protected:
  std::string       address_ = "tcp://localhost:5555";
  std::atomic<bool> alive_   ;
  std::future<void> future_  ;
};
}

#endif