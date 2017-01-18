#ifndef PLI_VIS_WINDOW_HPP_
#define PLI_VIS_WINDOW_HPP_

#include <memory>

#include <QMainWindow>

#include <hdf5/hdf5_io.hpp>

#include <ui_window.h>
#include <attributes/loggable.hpp>

namespace pli
{
class window : public QMainWindow, public Ui_window, public loggable<window>
{
public:
  window();

private:
  void bind_actions (); 
  void update_viewer() const;

  std::unique_ptr<pli::hdf5_io<float>> io_;

  std::array<std::size_t, 3> offset_            ;
  std::array<std::size_t, 3> size_              ;
  
  bool                       fom_show_          ;
  float                      fom_scale_         ;

  bool                       fdm_show_          ;
  std::array<std::size_t, 3> fdm_block_size_    ;
  std::array<std::size_t, 2> fdm_histogram_bins_;
  std::size_t                fdm_max_order_     ;
  std::array<std::size_t, 2> fdm_samples_       ;
};
}

#endif
