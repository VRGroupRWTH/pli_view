#ifndef ZERNIKE_FRAG_GLSL_
#define ZERNIKE_FRAG_GLSL_

#include <string>

namespace shaders
{
static std::string zernike_frag = R"(\
#version 450

layout(std430, binding = 0) readonly buffer Coefficients 
{
  float coefficients[];
};

in vertex_data 
{
  vec2 relative_position;
  uint coefficient_index;
} fs_in;

void main()
{
  // TODO
}
)";
}

#endif
