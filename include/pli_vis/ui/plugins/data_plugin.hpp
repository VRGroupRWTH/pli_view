#ifndef PLI_VIS_DATA_PLUGIN_HPP_
#define PLI_VIS_DATA_PLUGIN_HPP_

#include <future>

#include <vector_types.h>

#include <pli_vis/io/io.hpp>
#include <pli_vis/ui/plugin.hpp>
#include <ui_data_toolbox.h>

namespace pli
{
class data_plugin : public plugin<data_plugin, Ui_data_toolbox>
{
  Q_OBJECT

public:
  explicit data_plugin(QWidget* parent = nullptr);

  void start() override;
  
  const std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>& transmittance_bounds() const
  {
    return transmittance_bounds_;
  }
  const std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>& retardation_bounds  () const
  {
    return retardation_bounds_;
  }
  const std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>& direction_bounds    () const
  {
    return direction_bounds_;
  }
  const std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>& inclination_bounds  () const
  {
    return inclination_bounds_;
  }
  const std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>>& mask_bounds         () const
  {
    return mask_bounds_;
  }
  const std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>>& unit_vector_bounds  () const
  {
    return unit_vector_bounds_;
  }

  const boost::multi_array<float, 3>& transmittance() const
  {
    return transmittance_;
  }
  const boost::multi_array<float, 3>& retardation  () const
  {
    return retardation_;
  }
  const boost::multi_array<float, 3>& direction    () const
  {
    return direction_;
  }
  const boost::multi_array<float, 3>& inclination  () const
  {
    return inclination_;
  }
  const boost::multi_array<float, 3>& mask         () const
  {
    return mask_;
  }
  const boost::multi_array<float, 4>& unit_vector  () const
  {
    return unit_vector_;
  }
  
  boost::multi_array<unsigned char, 2> generate_preview_image(std::size_t y_resolution = 2048 );
  boost::multi_array<float3, 3>        generate_vectors      (bool        cartesian    = false);

signals:
  void on_change();
  void on_load  ();

private:
  io                io_    ;
  std::future<void> future_;
  
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> transmittance_bounds_;
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> retardation_bounds_  ;
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> direction_bounds_    ; 
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> inclination_bounds_  ;  
  std::pair<std::array<std::size_t, 3>, std::array<std::size_t, 3>> mask_bounds_         ;    
  std::pair<std::array<std::size_t, 4>, std::array<std::size_t, 4>> unit_vector_bounds_  ;

  boost::multi_array<float, 3> transmittance_;
  boost::multi_array<float, 3> retardation_  ;
  boost::multi_array<float, 3> direction_    ;
  boost::multi_array<float, 3> inclination_  ;
  boost::multi_array<float, 3> mask_         ;
  boost::multi_array<float, 4> unit_vector_  ;
};
}

#endif
