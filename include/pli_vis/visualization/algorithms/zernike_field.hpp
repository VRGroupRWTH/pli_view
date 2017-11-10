#ifndef PLI_VIS_ZERNIKE_FIELD_HPP_
#define PLI_VIS_ZERNIKE_FIELD_HPP_

#include <pli_vis/aspects/renderable.hpp>

namespace pli
{
class zernike_field : public renderable
{
public:
  void initialize()                     override;
  void render    (const camera* camera) override;
};
}

#endif