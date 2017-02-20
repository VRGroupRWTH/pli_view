#ifndef OPENGL_HPP_
#define OPENGL_HPP_

#ifdef _WIN32
#  include <gl/glew.h>
#elif __APPLE__
#  include <OpenGL/gl.h>
#else
#  include <gl/gl.h>
#endif

#endif