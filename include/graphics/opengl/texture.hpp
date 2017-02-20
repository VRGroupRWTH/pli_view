#ifndef TEXTURE_HPP_
#define TEXTURE_HPP_

#include "buffer.hpp"

namespace gl
{
template<GLenum target>
class texture
{
public:
   texture()
  {
    glGenTextures   (1, &id_);
  }
   texture(GLuint id) : id_(id), managed_(false)
  {

  }
  ~texture()
  {
    if (managed_)
      glDeleteTextures(1, &id_);
  }
  
  void        bind             ()
  {
    glBindTexture(target, id_);
  }
  void        unbind           ()
  {
    glBindTexture(target, 0  );
  }
                               
  bool        is_valid         () const
  {
    return glIsTexture(id_);
  }
                               
  GLuint      id               () const
  {
    return id_;
  }
                               
  static void set_active       (GLenum texture_unit)
  {
    glActiveTexture(texture_unit);
  }
  static void enable           ()
  {
    glEnable(target);
  }
  static void disable          ()
  {
    glDisable(target);
  }
                               
  static void min_filter       (GLenum mode)
  {
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, mode);
  }
  static void mag_filter       (GLenum mode)
  {
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, mode);
  }
                               
  static void wrap_s           (GLenum mode)
  {
    glTexParameteri(target, GL_TEXTURE_WRAP_S, mode);
  }
  static void wrap_t           (GLenum mode)
  {
    glTexParameteri(target, GL_TEXTURE_WRAP_T, mode);
  }
  static void wrap_r           (GLenum mode)
  {
    glTexParameteri(target, GL_TEXTURE_WRAP_R, mode);
  }

  static void generate_mipmaps ()
  {
    glGenerateMipmap(target);
  }

protected:
  GLuint id_      = 0;
  bool   managed_ = true;
};

class texture_1d : public texture<GL_TEXTURE_1D>
{
public:
  texture_1d()          : texture()   { }
  texture_1d(GLuint id) : texture(id) { }
  
  static void image_1d(GLenum internal_format, GLsizei width, GLenum format,  GLenum type, const void* data = nullptr)
  {
    glTexImage1D(GL_TEXTURE_1D, 0, internal_format, width, 0, format, type, data);
  }
};
class texture_2d : public texture<GL_TEXTURE_2D>
{
public:
  texture_2d()          : texture()   { }
  texture_2d(GLuint id) : texture(id) { }
 
  static void image_2d(GLenum internal_format, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* data = nullptr)
  {
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, data);
  }
};
class texture_3d : public texture<GL_TEXTURE_3D>
{
public:
  texture_3d()          : texture()   { }
  texture_3d(GLuint id) : texture(id) { }
 
  static void image_3d(GLenum internal_format, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* data = nullptr)
  {
    glTexImage3D(GL_TEXTURE_3D, 0, internal_format, width, height, depth, 0, format, type, data);
  }
};

class buffer_texture : public texture<GL_TEXTURE_BUFFER>
{
public:
  buffer_texture()          : texture()   { }
  buffer_texture(GLuint id) : texture(id) { }
 
  template<GLenum buffer_type>
  void set_buffer(GLenum internalFormat, const buffer<buffer_type>& buffer)
  {
     glTexBuffer(GL_TEXTURE_BUFFER, internalFormat, buffer.id());
  }
};
}

#endif
