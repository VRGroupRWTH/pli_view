#ifndef ZERNIKE_VERT_GLSL_
#define ZERNIKE_VERT_GLSL_

#include <string>

namespace shaders
{
static std::string zernike_vert = R"(\
#version 450
#extension GL_ARB_explicit_attrib_location : enable

layout(location = 0) in vec3  position     ;
layout(location = 1) in uvec2 sampling_size;

out vertex_data 
{
  vec3  position     ;
  uvec2 sampling_size;
} vs_out;

void main()
{
  gl_Position          = vec4(0.0)    ;
  vs_out.position      = position     ;
  vs_out.sampling_size = sampling_size;
}
)";
}


#endif