#ifndef BUFFER_HPP_
#define BUFFER_HPP_

namespace gl
{
template<GLenum target, GLenum usage = GL_STATIC_DRAW>
class buffer
{
public:
   buffer()
  {
    glGenBuffers(1, &id_);
  }
   buffer(GLuint id) : id_(id), managed_(false)
  {
    
  }
  ~buffer()
  {
    if (managed_)
      glDeleteBuffers(1, &id_);
  }

  void   bind         ()
  {
    glBindBuffer(target, id_);
  }
  void   unbind       ()
  {
    glBindBuffer(target, 0  );
  }

  void   bind_base    (GLuint index)
  {
    glBindBufferBase (target, index, id_);
  };
  void   bind_range   (GLuint index, GLintptr offset, GLsizei length)
  {
    glBindBufferRange(target, index, id_, offset, length);
  }

  void   set_data     (GLsizeiptr size, const void* data)
  {
    glBufferData(target, size, data, usage);
  }
  void   allocate     (GLsizeiptr size)
  {
    set_data(size, nullptr);
  }

  void   set_sub_data (GLintptr offset, GLsizeiptr size, const void* data)
  {
    glBufferSubData(target, offset, size, data);
  }
  void*  get_sub_data (GLintptr offset, GLsizeiptr size) const
  {
    void* subdata = nullptr;
    glGetBufferSubData(target, offset, size, subdata);
    return subdata;
  }

  void*  map          (GLenum access = GL_READ_WRITE)
  {
    return glMapBuffer(target, access);
  }
  void   unmap        ()
  {
    glUnmapBuffer(target);
  }

  bool   is_valid     () const
  {
    return glIsBuffer(id_);
  }

  GLuint id           () const
  {
    return id_;
  }
  
private:
  GLuint id_      = 0;
  bool   managed_ = true;
};

typedef buffer<GL_ARRAY_BUFFER>         array_buffer;
typedef buffer<GL_ELEMENT_ARRAY_BUFFER> index_buffer;
typedef buffer<GL_PIXEL_PACK_BUFFER>    pixel_pack_buffer;
typedef buffer<GL_PIXEL_UNPACK_BUFFER>  pixel_unpack_buffer;

typedef array_buffer vertex_buffer;
}

#endif