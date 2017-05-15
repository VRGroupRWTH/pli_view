#ifndef PLI_VIS_FDM_PLUGIN_HPP_
#define PLI_VIS_FDM_PLUGIN_HPP_

#include <future>

#include <cusolverDn.h>

#include <attributes/loggable.hpp>
#include <ui/plugins/plugin.hpp>
#include <ui_fdm_toolbox.h>

namespace pli
{
class odf_field;

class fdm_plugin : 
  public plugin, 
  public loggable<fdm_plugin>, 
  public Ui_fdm_toolbox
{
public:
  fdm_plugin(QWidget* parent = nullptr);
  void start  () override;
  void destroy() override;

private:
  void calculate         ();
  void extract_peaks     ();
  void set_visible_layers() const;

  float              threshold_multiplier_ = 0.01;
  odf_field*         odf_field_;
  std::future<void>  future_   ;
  cusolverDnHandle_t cusolver_ ;
  cublasHandle_t     cublas_   ;

};
}

#endif
