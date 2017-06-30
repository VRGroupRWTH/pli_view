#ifndef VOLUME_RENDERER_PREPASS_FRAG_GLSL_
#define VOLUME_RENDERER_PREPASS_FRAG_GLSL_

#include <string>

namespace shaders
{
std::string volume_renderer_prepass_frag = R"(\
#version 400

in  vec3 vert_color;
out vec4 frag_color;

void main()
{
  frag_color = vec4(vert_color, 1.0);
}
)";
}


#endif