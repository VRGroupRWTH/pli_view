#ifndef SIMPLE_TEXTURE_VERT_GLSL_
#define SIMPLE_TEXTURE_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string simple_texture_vert = R"(\
#version 400

uniform mat4 model         ;
uniform mat4 view          ;
uniform mat4 projection    ;
in      vec3 vertex        ;
in      vec2 texcoords     ;
out     vec2 vert_texcoords;

void main()
{
  gl_Position    = projection * view * model * vec4(vertex, 1.0);
  vert_texcoords = texcoords;
}
)";
}


#endif