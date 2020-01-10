#ifndef PLI_VIS_ODF_PLUGIN_HPP_
#define PLI_VIS_ODF_PLUGIN_HPP_

#include <future>

#include <cusolverDn.h>
#include <boost/multi_array.hpp>

#include <pli_vis/ui/plugin.hpp>
#include <ui_odf_toolbox.h>

namespace pli
{
class odf_field;

class odf_plugin : public plugin<odf_plugin, Ui_odf_toolbox>
{
public:
  explicit odf_plugin(QWidget* parent = nullptr);

  void start  () override;
  void destroy() override;

private:
  void calculate         ();
  void set_visible_layers() const;

  boost::multi_array<float , 4> coefficients_;
  odf_field*                    odf_field_   ;                              
  std::future<void>             future_      ;                                          
  cusolverDnHandle_t            cusolver_    ;
  cublasHandle_t                cublas_      ;

};
}

#endif
