#include /* implements */ <math/transform.hpp>

#include <algorithm>

namespace pli
{
transform::transform(const vec3f& translation, const quatf& rotation, const vec3f& scale)
  : translation_(translation), rotation_(rotation), scale_(scale)
{
  update_matrix();
}
transform::transform(const vec3f& translation, const vec3f& rotation_euler, const vec3f& scale)
  : transform(translation, quatf(radians(rotation_euler)), scale)
{

}

transform& transform::set_translation       (const vec3f& translation)
{
  translation_ = translation;
  update_matrix();
  return *this;
}
transform& transform::set_rotation          (const quatf& rotation)
{
  rotation_ = rotation;
  update_matrix();
  return *this;
}
transform& transform::set_rotation_euler    (const vec3f& rotation_euler)
{
  rotation_ = quatf(radians(rotation_euler));
  update_matrix();
  return *this;
}
transform& transform::set_scale             (const vec3f& scale)
{
  scale_ = scale;
  update_matrix();
  return *this;
}

transform& transform::translate             (const vec3f& amount)
{
  return set_translation(amount + translation_);
}
transform& transform::rotate                (const quatf& amount)
{
  return set_rotation   (amount * rotation_);
}
transform& transform::look_at               (const vec3f& target, const vec3f& up_vector)
{
  auto forward_vector   = normalize(target - translation_);
  auto right_vector     = normalize(cross(forward_vector, up_vector));
  auto true_up_vector   = normalize(cross(right_vector, forward_vector));

  auto forward_rotation = pli::rotation(                   vec3f(0, 0, 1), forward_vector);
  auto up_rotation      = pli::rotation(forward_rotation * vec3f(0, 1, 0), true_up_vector);

  return set_rotation(up_rotation * forward_rotation);
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
  matrix_ = pli::translate(translation_) * mat4_cast(rotation_) * pli::scale(scale_);
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
