#ifndef PLI_VIS_CAMERA_HPP_
#define PLI_VIS_CAMERA_HPP_

#include <glm/glm.hpp>

#include <pli_vis/visualization/primitives/transform.hpp>

namespace pli
{
/// A camera is a view transform with an additional projection matrix.
/// Supports perspective and orthographic projection.
class camera : public transform
{
public:
  camera();

  bool             orthographic             () const { return orthographic_     ; }
  float	           near_clip_plane          () const { return near_clip_plane_  ; }
  float	           far_clip_plane           () const { return far_clip_plane_   ; }
  float	           aspect_ratio             () const { return aspect_ratio_     ; }
  float	           vertical_fov             () const { return vertical_fov_     ; }
  float	           orthographic_size        () const { return orthographic_size_; }
  
  const glm::mat4& projection_matrix        () const { return projection_matrix_; }
        glm::mat4  view_projection_matrix   () const { return projection_matrix_ * inverse_absolute_matrix(); }

  void             set_orthographic         (bool  orthographic     );
  void             set_near_clip_plane      (float near_clip_plane  );
  void             set_far_clip_plane       (float far_clip_plane   );
  void             set_aspect_ratio         (float aspect_ratio     );
  void             set_vertical_fov         (float vertical_fov     );
  void             set_orthographic_size    (float orthographic_size);

private:
  void             update_projection_matrix ();

  /// Toggle for orthographic / perspective. Default perspective.
  bool  orthographic_      = false;
 
  /// Shared parameters.
  float	near_clip_plane_   = 0.01F;
  float	far_clip_plane_    = 10000.0F;
  float	aspect_ratio_      = 4.0F / 3.0F;

  /// When camera is perspective , camera's viewing volume is defined by vertical fov.
  float	vertical_fov_      = glm::radians(68.0F);
  
  /// When camera is orthographic, camera's viewing volume is defined by orthographic size.
  float	orthographic_size_ = 100;

  glm::mat4 projection_matrix_;
};
}

#endif
