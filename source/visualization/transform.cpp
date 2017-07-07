#include <pli_vis/visualization/transform.hpp>

#include <algorithm>

#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace pli
{
transform::transform(const glm::vec3& translation, const glm::quat& rotation, const glm::vec3& scale)
  : translation_(translation), rotation_(rotation), scale_(scale)
{
  update_matrix();
}
transform::transform(const glm::vec3& translation, const glm::vec3& rotation_euler, const glm::vec3& scale)
  : transform(translation, glm::quat(glm::radians(rotation_euler)), scale)
{

}

transform& transform::set_translation       (const glm::vec3& translation)
{
  translation_ = translation;
  update_matrix();
  return *this;
}
transform& transform::set_rotation          (const glm::quat& rotation)
{
  rotation_ = rotation;
  update_matrix();
  return *this;
}
transform& transform::set_rotation_euler    (const glm::vec3& rotation_euler)
{
  rotation_ = glm::quat(radians(rotation_euler));
  update_matrix();
  return *this;
}
transform& transform::set_scale             (const glm::vec3& scale)
{
  scale_ = scale;
  update_matrix();
  return *this;
}

transform& transform::translate             (const glm::vec3& amount)
{
  return set_translation(amount + translation_);
}
transform& transform::rotate                (const glm::quat& amount)
{
  return set_rotation   (amount * rotation_);
}
transform& transform::look_at               (const glm::vec3& target, const glm::vec3& up_vector)
{
  return set_rotation(glm::conjugate(glm::toQuat(lookAt(translation_, target, up_vector))));
}

void       transform::set_parent            (transform* parent)
{
  if (parent_ != nullptr)
    parent_->children_.erase(std::remove(parent_->children_.begin(), 
                                         parent_->children_.end  (), this), 
                                         parent_->children_.end  ());

  parent_ = parent;

  if (parent_ != nullptr)
    parent_->children_.push_back(this);

  update_absolute_matrix();
}
transform* transform::child                 (std::size_t index) const
{
  if (children_.size() - 1 < index)
    return nullptr;
  return children_[index];
}

void       transform::update_matrix         ()
{
  matrix_ = glm::translate(translation_) * mat4_cast(rotation_) * glm::scale(scale_);
  // A change of local matrix implies a change of world matrix, hence:
  update_absolute_matrix();
}
void       transform::update_absolute_matrix()
{
  absolute_matrix_ = parent_ ? parent_->absolute_matrix_ * matrix_ : matrix_;
  // Propagate the change to hierarchy.
  for (auto child : children_)
    child->update_absolute_matrix();
}
}
