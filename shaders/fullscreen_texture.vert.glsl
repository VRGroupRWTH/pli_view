#ifndef FULLSCREEN_TEXTURE_VERT_GLSL_
#define FULLSCREEN_TEXTURE_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string fullscreen_texture_vert = R"(\
#version 450

layout(location = 0) in vec3 position ;
layout(location = 1) in vec2 texcoords;

layout(location = 0) out vertex_data 
{
  vec2 texcoords;
} vs_out;

void main()
{
  gl_Position      = vec4(position, 1.0);
  vs_out.texcoords = texcoords;
}
)";
}

#endif