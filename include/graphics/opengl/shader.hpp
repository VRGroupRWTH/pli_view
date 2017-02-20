#ifndef SHADER_HPP_
#define SHADER_HPP_

#include <fstream>
#include <string>

namespace gl
{
template<GLenum type>
class shader
{
public:
   shader()
  {
    id_ = glCreateShader(type);
  }
   shader(GLuint id) : id_(id), managed_(false)
  {

  }
   shader(const std::string& filename, bool compile = true) : shader()
  {
    from_file(filename);
    if (compile)
      compile();
  }
  ~shader()
  {
    if (managed_)
      glDeleteShader(id_);
  }

  void   from_file   (const std::string& filename)
  {
    std::ifstream filestream(filename);
    from_string(std::string(std::istreambuf_iterator<char>(filestream), std::istreambuf_iterator<char>()));
  }
  void   from_string (const std::string& shader_string)
  {
    auto shader_cstring = shader_string.c_str();
    glShaderSource(id_, 1, &shader_cstring, nullptr);
  }

  bool   compile     ()
  {
    glCompileShader(id_);

    auto status = 0;
    glGetShaderiv(id_, GL_COMPILE_STATUS, &status);
    return status;
  }

  bool   is_valid    () const
  {
    return glIsShader(id_);
  }

  GLuint id          () const
  {
    return id_;
  }

private:
  GLuint id_      = 0;
  bool   managed_ = true;
};

typedef shader<GL_VERTEX_SHADER>          vertex_shader;
typedef shader<GL_FRAGMENT_SHADER>        fragment_shader;
typedef shader<GL_GEOMETRY_SHADER>        geometry_shader;
typedef shader<GL_TESS_CONTROL_SHADER>    tess_control_shader;
typedef shader<GL_TESS_EVALUATION_SHADER> tess_evaluation_shader;
typedef shader<GL_COMPUTE_SHADER>         compute_shader;
}
#endif
