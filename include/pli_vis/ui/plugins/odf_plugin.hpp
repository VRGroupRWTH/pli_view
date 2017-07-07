#ifndef PLI_VIS_FDM_PLUGIN_HPP_
#define PLI_VIS_FDM_PLUGIN_HPP_

#include <future>

#include <cusolverDn.h>
#include <boost/multi_array.hpp>

#include <pli_vis/aspects/loggable.hpp>
#include <pli_vis/ui/plugin.hpp>

#include <ui_odf_toolbox.h>

namespace pli
{
class odf_field;

class odf_plugin : public plugin, public loggable<odf_plugin>, public Ui_odf_toolbox
{
public:
  odf_plugin(QWidget* parent = nullptr);
  void start  () override;
  void destroy() override;

private:
  void calculate         ();
  void extract_peaks     ();
  void set_visible_layers() const;

  float                        threshold_multiplier_ = 0.01F;
  boost::multi_array<float, 4> coefficients_;
  odf_field*                   odf_field_   ;                              
  std::future<void>            future_      ;                                          
  cusolverDnHandle_t           cusolver_    ;
  cublasHandle_t               cublas_      ;

};
}

#endif
