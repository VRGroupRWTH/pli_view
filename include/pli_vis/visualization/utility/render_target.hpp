#ifndef PLI_VIS_RENDER_TARGET_HPP_
#define PLI_VIS_RENDER_TARGET_HPP_

#include <glm/glm.hpp>

#include <pli_vis/opengl/opengl.hpp>
#include <pli_vis/opengl/framebuffer.hpp>
#include <pli_vis/opengl/texture.hpp>

namespace pli
{
class render_target
{
public:
  enum class mode
  {
    color_only,
    depth_only,
    color_and_depth
  };

  render_target(const glm::uvec2& size, mode mode = mode::color_and_depth);
  render_target(const render_target&  that) = default;
  render_target(      render_target&& temp) = default;
  virtual ~render_target()                  = default;

  render_target& operator=(const render_target&  that) = default;
  render_target& operator=(      render_target&& temp) = default;

  void resize(const glm::uvec2& size );
  void bind  ();
  void unbind();

  const gl::framebuffer& framebuffer  () const;
  const gl::texture_2d&  color_texture() const;
  const gl::texture_2d&  depth_texture() const;

protected:
  mode            mode_            ;
  gl::framebuffer last_framebuffer_;
  gl::framebuffer framebuffer_     ;
  gl::texture_2d  color_texture_   ;
  gl::texture_2d  depth_texture_   ;
};
}

#endif
