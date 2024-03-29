#ifndef SIMPLE_COLOR_TEXTURE_VERT_GLSL_
#define SIMPLE_COLOR_TEXTURE_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string simple_color_texture_vert = R"(\
#version 400

uniform mat4 model         ;
uniform mat4 view          ;
uniform mat4 projection    ;
uniform vec2 size          ;
in      vec3 position      ;
in      vec2 texcoords     ;
out     vec2 vert_texcoords;

void main()
{
  gl_Position    = projection * view * model * vec4(2.0f * (position.x - 0.5f) * size.x - 0.5f, 2.0f * (position.y - 0.5f) * size.y - 0.5f, position.z, 1.0);
  vert_texcoords = texcoords;
}
)";
}

#endif
