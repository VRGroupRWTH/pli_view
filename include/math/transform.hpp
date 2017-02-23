#ifndef PLI_VIS_TRANSFORM_HPP_
#define PLI_VIS_TRANSFORM_HPP_

#include <vector>

#include "linear_math.hpp"

namespace pli
{
/// Represents a transformation in 3D space. Supports hierarchies.
class transform
{
public:
  transform(const vec3f& translation                 , const quatf& rotation       = quatf(0, 0, 0, 1), const vec3f& scale = vec3f(1, 1, 1));
  transform(const vec3f& translation = vec3f(0, 0, 0), const vec3f& rotation_euler = vec3f(0, 0, 0)   , const vec3f& scale = vec3f(1, 1, 1));
  
  const vec3f& translation             () const { return translation_                    ;}
  const quatf& rotation                () const { return rotation_                       ;}
        vec3f  rotation_euler          () const { return degrees(eulerAngles(rotation_)) ;}
  const vec3f& scale                   () const { return scale_                          ;}
                                       
  const mat4f& matrix                  () const { return matrix_                         ;}
        mat4f  inverse_matrix          () const { return inverse(matrix_)                ;}
  const mat4f& absolute_matrix         () const { return absolute_matrix_                ;}
        mat4f  inverse_absolute_matrix () const { return inverse(absolute_matrix_)       ;}
                                       
        vec3f  forward                 () const { return rotation_ * vec3f(0, 0, 1)      ;}
        vec3f  up                      () const { return rotation_ * vec3f(0, 1, 0)      ;}
        vec3f  right                   () const { return rotation_ * vec3f(1, 0, 0)      ;}
                                       
  transform*   parent                  () const { return parent_                         ;}
  std::size_t  child_count             () const { return children_.size()                ;}
                                       
  transform&   set_translation         (const vec3f& translation   );
  transform&   set_rotation            (const quatf& rotation      );
  transform&   set_rotation_euler      (const vec3f& rotation_euler);
  transform&   set_scale               (const vec3f& scale         );
                                       
  transform&   translate               (const vec3f& amount);
  transform&   rotate                  (const quatf& amount);
  transform&   look_at                 (const vec3f& target, const vec3f& up_vector = vec3f(0, 1, 0));
                                       
  void         set_parent              (transform*  parent);
  transform*   child                   (std::size_t index ) const;

protected:
  void         update_matrix           ();
  void         update_absolute_matrix  ();

  vec3f                   translation_;   
  quatf                   rotation_;      
  vec3f                   scale_;         

  mat4f                   matrix_;        
  mat4f                   absolute_matrix_;

  transform*              parent_ = nullptr;
  std::vector<transform*> children_;
};
}

#endif
