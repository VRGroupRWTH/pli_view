#ifndef GL_OPENGL_HPP_
#define GL_OPENGL_HPP_

#include <iostream>

#ifdef _WIN32
#  include <gl/glew.h>
#elif __APPLE__
#  include <OpenGL/gl.h>
#else
#  include <gl/gl.h>
#endif

namespace opengl
{
inline void init()
{
  #ifdef _WIN32
    glewExperimental = true;
    glewInit();
  #elif __APPLE__

  #else
  
  #endif
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