#include <pli_vis/visualization/utility/render_target.hpp>

namespace pli
{
render_target::render_target(const glm::uvec2& size, mode mode) : mode_(mode)
{
  bind();
  if(mode_ == mode::color_only || mode_ == mode::color_and_depth)
  {
    color_texture_.bind       ();
    color_texture_.wrap_s     (GL_CLAMP_TO_EDGE);
    color_texture_.wrap_t     (GL_CLAMP_TO_EDGE);
    color_texture_.min_filter (GL_NEAREST);
    color_texture_.mag_filter (GL_NEAREST);
    color_texture_.set_image  (GL_RGBA32F, size[0], size[1], GL_RGBA, GL_FLOAT);
    color_texture_.unbind     ();
    framebuffer_  .set_texture(GL_COLOR_ATTACHMENT0, color_texture_);
  }
  if(mode_ == mode::depth_only || mode_ == mode::color_and_depth)
  {
    depth_texture_.bind       ();
    depth_texture_.wrap_s     (GL_CLAMP_TO_EDGE);
    depth_texture_.wrap_t     (GL_CLAMP_TO_EDGE);
    depth_texture_.min_filter (GL_NEAREST);
    depth_texture_.mag_filter (GL_NEAREST);
    depth_texture_.set_image  (GL_DEPTH_COMPONENT32F, size[0], size[1], GL_DEPTH_COMPONENT, GL_FLOAT);
    depth_texture_.unbind     ();
    framebuffer_  .set_texture(GL_DEPTH_ATTACHMENT, depth_texture_);
  }
  assert(framebuffer_.is_valid() && framebuffer_.is_complete());
  unbind();
}

void render_target::resize(const glm::uvec2& size)
{
  if (mode_ == mode::color_only || mode_ == mode::color_and_depth)
  {
    color_texture_.bind     ();
    color_texture_.set_image(GL_RGBA32F, size[0], size[1], GL_RGBA, GL_FLOAT);
    color_texture_.unbind   ();
  }
  if (mode_ == mode::depth_only || mode_ == mode::color_and_depth)
  {
    depth_texture_.bind     ();
    depth_texture_.set_image(GL_DEPTH_COMPONENT32F, size[0], size[1], GL_DEPTH_COMPONENT, GL_FLOAT);
    depth_texture_.unbind   ();
  }
}
void render_target::bind  ()
{
  GLint last_framebuffer_id;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &last_framebuffer_id);
  last_framebuffer_ = gl::framebuffer(last_framebuffer_id);

  framebuffer_.bind();
}
void render_target::unbind()
{
  last_framebuffer_.bind();
}

gl::framebuffer* render_target::framebuffer  ()
{
  return &framebuffer_;
}
gl::texture_2d*  render_target::color_texture()
{
  return &color_texture_;
}
gl::texture_2d*  render_target::depth_texture()
{
  return &depth_texture_;
}
}
