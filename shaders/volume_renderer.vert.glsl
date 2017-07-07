#ifndef VOLUME_RENDERER_VERT_GLSL_
#define VOLUME_RENDERER_VERT_GLSL_

#include <string>

namespace shaders
{
std::string volume_renderer_vert = R"(\
#version 400

uniform mat4 projection;
uniform mat4 view      ;
in      vec3 vertex    ;
out     vec3 vert_color;

void main()
{
  vert_color  = vertex;
  gl_Position = projection * view * vec4(vertex, 1.0);
}
)";
}


#endif