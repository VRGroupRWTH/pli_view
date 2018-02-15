#ifndef FULLSCREEN_TEXTURE_VERT_GLSL_
#define FULLSCREEN_TEXTURE_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string fullscreen_texture_vert = R"(\
#version 400

in  vec3 position      ;
in  vec2 texcoords     ;
out vec2 vert_texcoords;

void main()
{
  gl_Position    = vec4(position, 1.0);
  vert_texcoords = texcoords;
}
)";
}

#endif