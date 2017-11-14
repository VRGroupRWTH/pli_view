#ifndef GL_PROGRAM_HPP_
#define GL_PROGRAM_HPP_

#include <vector>

#include "shader.hpp"

namespace gl
{
class program
{
public:
   program()
  {
    id_ = glCreateProgram();
  }
   program(GLuint id) : id_(id), managed_(false)
  {
    
  }
  ~program()
  {
    if (managed_)
      glDeleteProgram(id_);
  }

  template<GLenum type>
  void   attach_shader                (const shader<type>& shader)
  {
    glAttachShader(id_, shader.id());
  }
  void   detach_shaders               ()
  {
    GLsizei count;
    std::vector<GLuint> shaders(32);
    glGetAttachedShaders(id_, (GLsizei)shaders.size(), &count, shaders.data());
    for (auto i = 0; i < count; i++)
      glDetachShader(id_, shaders[i]);
  }
  bool   link                         ()
  {
    glLinkProgram(id_);

    auto status = 0;
    glGetProgramiv(id_, GL_LINK_STATUS, &status);
    return status != 0;
  }
                                      
  void   bind                         ()
  {
    glUseProgram(id_);
  }
  void   unbind                       ()
  {
    glUseProgram(0  );
  }

  void   set_attribute_buffer         (const std::string& name, GLuint size, GLuint type, bool normalize = true, GLuint stride = 0, GLuint offset = 0)
  {
    auto location = get_attribute_location(name);
    if (location < 0)
      return;
    glVertexAttribPointer(location, size, type, normalize, stride, reinterpret_cast<GLvoid*>(std::size_t(offset)));
  }
  void   set_attribute_buffer_integer (const std::string& name, GLuint size, GLuint type,                        GLuint stride = 0, GLuint offset = 0)
  {
    auto location = get_attribute_location(name);
    if (location < 0)
      return;
    glVertexAttribIPointer(location, size, type, stride, reinterpret_cast<GLvoid*>(std::size_t(offset)));

  }
  void   set_attribute_buffer_double  (const std::string& name, GLuint size, GLuint type,                        GLuint stride = 0, GLuint offset = 0)
  {
    auto location = get_attribute_location(name);
    if (location < 0)
      return;
    glVertexAttribLPointer(location, size, type, stride, reinterpret_cast<GLvoid*>(std::size_t(offset)));
  }
                                      
  void   enable_attribute_array       (const std::string& name)
  {
    auto location = get_attribute_location(name);
    if (location < 0)
      return;
    glEnableVertexAttribArray(location);
  }
  void   disable_attribute_array      (const std::string& name)
  {
    auto location = get_attribute_location(name);
    if (location < 0)
      return;
    glDisableVertexAttribArray(location);
  }
                                      
  GLint  get_uniform_location         (const std::string& name)
  {
    return glGetUniformLocation(id_, name.c_str());
  }
  GLint  get_attribute_location       (const std::string& name)
  {
    return glGetAttribLocation(id_, name.c_str());
  }
                                      
  bool   is_valid                     () const
  {
    return glIsProgram(id_) != 0;
  }
                                      
  GLuint id                           () const
  {
    return id_;
  }
                                      
  template<typename Type>             
  void   set_uniform                  (const std::string& name, const Type&              value ) { }
  template<typename Type>             
  void   set_uniform_array            (const std::string& name, const std::vector<Type>& values) { }

private:
  GLuint id_      = 0;
  bool   managed_ = true;
};

#define SPECIALIZE_SET_UNIFORM_BASIC_TYPES(TYPE, GL_POSTFIX) \
template <> \
inline void program::set_uniform \
(const std::string& name, const TYPE& value) \
{ \
  auto location = get_uniform_location(name); \
  if (location < 0) \
    return; \
  glUniform##GL_POSTFIX(location, value); \
} \
template <> \
inline void program::set_uniform_array \
(const std::string& name, const std::vector<TYPE>& values) \
{ \
  auto location = get_uniform_location(name); \
  if (location < 0) \
    return; \
  glUniform##GL_POSTFIX##v(location, (GLsizei) values.size(), values.data()); \
} \

SPECIALIZE_SET_UNIFORM_BASIC_TYPES(float       , 1f )
SPECIALIZE_SET_UNIFORM_BASIC_TYPES(int         , 1i )
SPECIALIZE_SET_UNIFORM_BASIC_TYPES(unsigned int, 1ui)

#undef SPECIALIZE_SET_UNIFORM_BASIC_TYPES

// Boolean is a special case (std::vector<bool> does not have a .data() function).
template <>
inline void program::set_uniform(const std::string& name, const bool& value)
{
  auto location = get_uniform_location(name);
  if (location < 0)
    return;
  glUniform1i(location, value);
}

}

#endif
