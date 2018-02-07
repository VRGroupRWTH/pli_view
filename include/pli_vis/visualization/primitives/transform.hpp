#ifndef PLI_VIS_TRANSFORM_HPP_
#define PLI_VIS_TRANSFORM_HPP_

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <vector>

namespace pli
{
/// Represents a transformation in 3D space. Supports hierarchies.
class transform
{
public:
  transform(const glm::vec3& translation                     , const glm::quat& rotation       = glm::quat(0, 0, 0, 1), const glm::vec3& scale = glm::vec3(1, 1, 1));
  transform(const glm::vec3& translation = glm::vec3(0, 0, 0), const glm::vec3& rotation_euler = glm::vec3(0, 0, 0)   , const glm::vec3& scale = glm::vec3(1, 1, 1));
  
  const glm::vec3& translation             () const { return translation_                    ;}
  const glm::quat& rotation                () const { return rotation_                       ;}
        glm::vec3  rotation_euler          () const { return glm::degrees(glm::eulerAngles(rotation_)) ;}
  const glm::vec3& scale                   () const { return scale_                          ;}
                                       
  const glm::mat4& matrix                  () const { return matrix_                         ;}
        glm::mat4  inverse_matrix          () const { return glm::inverse(matrix_)                ;}
  const glm::mat4& absolute_matrix         () const { return absolute_matrix_                ;}
        glm::mat4  inverse_absolute_matrix () const { return glm::inverse(absolute_matrix_)       ;}
                                       
        glm::vec3  forward                 () const { return rotation_ * glm::vec3(0, 0, 1)      ;}
        glm::vec3  up                      () const { return rotation_ * glm::vec3(0, 1, 0)      ;}
        glm::vec3  right                   () const { return rotation_ * glm::vec3(1, 0, 0)      ;}
                                       
  transform*       parent                  () const { return parent_                         ;}
  std::size_t      child_count             () const { return children_.size()                ;}
                                           
  transform&       set_translation         (const glm::vec3& translation   );
  transform&       set_rotation            (const glm::quat& rotation      );
  transform&       set_rotation_euler      (const glm::vec3& rotation_euler);
  transform&       set_scale               (const glm::vec3& scale         );
                                           
  transform&       translate               (const glm::vec3& amount);
  transform&       rotate                  (const glm::quat& amount);
  transform&       look_at                 (const glm::vec3& target, const glm::vec3& up_vector = glm::vec3(0, 1, 0));
                                           
  void             set_parent              (transform*  parent);
  transform*       child                   (std::size_t index ) const;

protected:
  void             update_matrix           ();
  void             update_absolute_matrix  ();

  glm::vec3               translation_;
  glm::quat               rotation_;      
  glm::vec3               scale_;
                          
  glm::mat4               matrix_;        
  glm::mat4               absolute_matrix_;
                          
  transform*              parent_ = nullptr;
  std::vector<transform*> children_;
};
}

#endif
