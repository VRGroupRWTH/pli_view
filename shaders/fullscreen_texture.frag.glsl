#ifndef FULLSCREEN_TEXTURE_FRAG_GLSL_
#define FULLSCREEN_TEXTURE_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string fullscreen_texture_frag = R"(\
#version 450

uniform sampler2D texture_unit;

in vertex_data 
{
  vec2 texcoords;
} fs_in;

layout(location = 0) out vec4 color;

void main()
{
  color = texture(texture_unit, fs_in.texcoords);
}
)";
}

#endif
