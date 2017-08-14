#ifndef LINEAO_ZOOM_PASS_FRAG_GLSL_
#define LINEAO_ZOOM_PASS_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string lineao_zoom_pass_frag = R"(\
#version 400

uniform mat4  model     ;
uniform mat4  view      ;
uniform mat4  projection;
flat in float vert_zoom ;
out     vec4  frag_color;

void main()
{
  frag_color = vec4(vert_zoom, vert_zoom, vert_zoom, vert_zoom);
}
)";
}

#endif
