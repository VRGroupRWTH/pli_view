#ifndef GL_FRAMEBUFFER_HPP_
#define GL_FRAMEBUFFER_HPP_

#include <vector>

#include "texture.hpp"

namespace gl
{
template<GLenum target>
class framebuffer_base
{
public:
   framebuffer_base()
  {
    glGenFramebuffers(1, &id_);
  }
   framebuffer_base(GLuint id) : id_(id), managed_(false)
  {

  }
  ~framebuffer_base()
  {
    if (managed_)
      glDeleteFramebuffers(1, &id_);
  }

  void        bind                 ()
  {
    glBindFramebuffer(target, id_);
  }
  void        unbind               ()
  {
    glBindFramebuffer(target, 0  );
  }

  template<GLenum type>
  typename std::enable_if<type == GL_TEXTURE_1D>::type set_texture(GLenum attachment, const texture<type>& texture, GLuint level = 0)
  {
    glFramebufferTexture1D(target, attachment, type, texture.id(), level);
  }
  template<GLenum type>
  typename std::enable_if<type == GL_TEXTURE_2D>::type set_texture(GLenum attachment, const texture<type>& texture, GLuint level = 0)
  {
    glFramebufferTexture2D(target, attachment, type, texture.id(), level);
  }
  template<GLenum type>
  typename std::enable_if<type == GL_TEXTURE_3D>::type set_texture(GLenum attachment, const texture<type>& texture, GLuint level = 0, GLuint layer = 0)
  {
    glFramebufferTexture3D(target, attachment, type, texture.id(), level, layer);
  }
                                   
  bool        is_complete          ()
  {
    return glCheckFramebufferStatus(target) == GL_FRAMEBUFFER_COMPLETE;
  }
                                   
  bool        is_valid             () const
  {
    return glIsFramebuffer(id_);
  }
                                   
  GLuint      id                   () const
  {
    return id_;
  }

  static void set_read_buffer      (GLenum attachment)
  {
    glReadBuffer(attachment);
  }
  static void set_draw_buffer      (GLenum attachment)
  {
    glDrawBuffers(1, &attachment);
  }
  static void set_draw_buffers     (const std::vector<GLenum> attachments)
  {
    glDrawBuffers(attachments.size(), attachments.data());
  }
  
  static void clear_color_buffer   (GLint draw_buffer_index, GLint   value)
  {
    glClearBufferiv (GL_COLOR, draw_buffer_index, &value);
  }
  static void clear_color_buffer   (GLint draw_buffer_index, GLuint  value)
  {
    glClearBufferuiv(GL_COLOR, draw_buffer_index, &value);
  }
  static void clear_color_buffer   (GLint draw_buffer_index, GLfloat value)
  {
    glClearBufferfv (GL_COLOR, draw_buffer_index, &value);
  }
  static void clear_color_buffer   (GLint draw_buffer_index, std::vector<GLint>   value)
  {
    glClearBufferiv (GL_COLOR, draw_buffer_index, value.data());
  }
  static void clear_color_buffer   (GLint draw_buffer_index, std::vector<GLuint>  value)
  {
    glClearBufferuiv(GL_COLOR, draw_buffer_index, value.data());
  }
  static void clear_color_buffer   (GLint draw_buffer_index, std::vector<GLfloat> value)
  {
    glClearBufferfv (GL_COLOR, draw_buffer_index, value.data());
  }
  static void clear_depth_buffer   (GLfloat value = 0)
  {
    glClearBufferfv(GL_DEPTH, 0, &value);
  }
  static void clear_stencil_buffer (GLint   value = 0)
  {
    const unsigned int clearValue = 0;
    glClearBufferiv(GL_STENCIL, 0, &value);
  }

private:
  GLuint id_      = 0;
  bool   managed_ = true;
};

typedef framebuffer_base<GL_FRAMEBUFFER>      framebuffer;
typedef framebuffer_base<GL_READ_FRAMEBUFFER> read_framebuffer;
typedef framebuffer_base<GL_DRAW_FRAMEBUFFER> draw_framebuffer;
}

#endif
