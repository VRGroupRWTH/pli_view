#ifndef PLI_VIS_LINEAR_MATH_HPP_
#define PLI_VIS_LINEAR_MATH_HPP_

#include <third_party/glm/glm.hpp>
#include <third_party/glm/gtc/quaternion.hpp>
#include <third_party/glm/gtx/transform.hpp>
#include <third_party/glm/gtx/quaternion.hpp>

namespace pli
{
using vec2f  = glm::fvec2;
using vec3f  = glm::fvec3;
using vec4f  = glm::fvec4;
using vec2d  = glm::dvec2;
using vec3d  = glm::dvec3;
using vec4d  = glm::dvec4;
using vec2i  = glm::ivec2;
using vec3i  = glm::ivec3;
using vec4i  = glm::ivec4;
using vec2ui = glm::uvec2;
using vec3ui = glm::uvec3;
using vec4ui = glm::uvec4;

using mat2f  = glm::fmat2;
using mat3f  = glm::fmat3;
using mat4f  = glm::fmat4;
using mat2d  = glm::dmat2;
using mat3d  = glm::dmat3;
using mat4d  = glm::dmat4;
             
using quatf  = glm::fquat;
using quatd  = glm::dquat;

using glm::angleAxis;
using glm::eulerAngles;
using glm::inverse;
using glm::length;
using glm::lerp;
using glm::lookAt;
using glm::mat4_cast;
using glm::rotation;

using glm::degrees;
using glm::radians;

using glm::translate;
using glm::rotate;
using glm::scale;

using glm::ortho;
using glm::perspective;
}

#endif
