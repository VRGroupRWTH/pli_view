#include <pli_vis/visualization/camera.hpp>

#include <glm/gtx/transform.hpp>

namespace pli
{
camera::camera() : transform()
{
  update_projection_matrix();
}

void camera::set_orthographic        (bool  orthographic     )
{
  orthographic_ = orthographic;
  update_projection_matrix();
}
void camera::set_near_clip_plane     (float near_clip_plane  )
{
  near_clip_plane_ = near_clip_plane;
  update_projection_matrix();
}
void camera::set_far_clip_plane      (float far_clip_plane   )
{
  far_clip_plane_ = far_clip_plane;
  update_projection_matrix();
}
void camera::set_aspect_ratio        (float aspect_ratio     )
{
  aspect_ratio_ = aspect_ratio;
  update_projection_matrix();
}
void camera::set_vertical_fov        (float vertical_fov     )
{
  vertical_fov_ = vertical_fov;
  update_projection_matrix();
}
void camera::set_orthographic_size   (float orthographic_size)
{
  orthographic_size_ = orthographic_size;
  update_projection_matrix();
}

void camera::update_projection_matrix()
{
  if (orthographic_)
  {
    auto ortho_half_height = orthographic_size_ * 2.0F;
    auto ortho_half_width  = ortho_half_height * aspect_ratio_;
    projection_matrix_ = glm::ortho(-ortho_half_width , 
                                     ortho_half_width , 
                                    -ortho_half_height, 
                                     ortho_half_height, 
                                     near_clip_plane_ , 
                                     far_clip_plane_  );
  }
  else
  {
    projection_matrix_ = glm::perspective(vertical_fov_   , 
                                          aspect_ratio_   , 
                                          near_clip_plane_, 
                                          far_clip_plane_ );
  }
}
}
