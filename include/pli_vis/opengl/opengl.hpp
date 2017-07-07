#ifndef GL_OPENGL_HPP_
#define GL_OPENGL_HPP_

#define GLEW_STATIC

#include <iostream>

#include <glew/GL/glew.h>

namespace opengl
{
inline void init()
{
  glewExperimental = true;
  glewInit();
}

inline void print_error(const char* prefix)
{
  GLenum error = GL_NO_ERROR;
  do 
  {
    error = glGetError();
    if (error != GL_NO_ERROR)
      std::cout << prefix << ": " << error << std::endl;
  } 
  while (error != GL_NO_ERROR);
}
}

#endif