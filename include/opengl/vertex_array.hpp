#ifndef GL_VERTEX_ARRAY_HPP_
#define GL_VERTEX_ARRAY_HPP_

namespace gl
{
class vertex_array
{
public:
   vertex_array()
  {
    glGenVertexArrays(1, &id_);
  }
   vertex_array(GLuint id) : id_(id), managed_(false)
  {
    
  }
  ~vertex_array()
  {
    if (managed_)
      glDeleteVertexArrays(1, &id_);
  }

  void   bind     ()
  {
    glBindVertexArray(id_);
  }
  void   unbind   ()
  {
    glBindVertexArray(0  );
  }

  bool   is_valid () const
  {
    return glIsVertexArray(id_);
  }

  GLuint id       () const
  {
    return id_;
  }

private:
  GLuint id_      = 0;
  bool   managed_ = true;
};
}

#endif
