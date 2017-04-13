#ifndef GL_GLM_UNIFORMS_HPP_
#define GL_GLM_UNIFORMS_HPP_

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "../program.hpp"

namespace gl
{
#define SPECIALIZE_SET_UNIFORM_VECTORS(TYPE, GL_POSTFIX) \
template <> \
inline void program::set_uniform       (const std::string& name, const TYPE& value) \
{ \
  auto location = get_uniform_location(name); \
  if (location < 0) \
    return; \
  glUniform##GL_POSTFIX##v(location, 1, glm::value_ptr(value)); \
} \
template <> \
inline void program::set_uniform_array (const std::string& name, const std::vector<TYPE>& values) \
{ \
  auto location = get_uniform_location(name); \
  if (location < 0) \
    return; \
  glUniform##GL_POSTFIX##v(location, (GLsizei) values.size(), glm::value_ptr(values[0])); \
} \

SPECIALIZE_SET_UNIFORM_VECTORS(glm::fvec2, 2f)
SPECIALIZE_SET_UNIFORM_VECTORS(glm::ivec2, 2i)
SPECIALIZE_SET_UNIFORM_VECTORS(glm::uvec2, 2ui)
SPECIALIZE_SET_UNIFORM_VECTORS(glm::fvec3, 3f)
SPECIALIZE_SET_UNIFORM_VECTORS(glm::ivec3, 3i)
SPECIALIZE_SET_UNIFORM_VECTORS(glm::uvec3, 3ui)
SPECIALIZE_SET_UNIFORM_VECTORS(glm::fvec4, 4f)
SPECIALIZE_SET_UNIFORM_VECTORS(glm::ivec4, 4i)
SPECIALIZE_SET_UNIFORM_VECTORS(glm::uvec4, 4ui)

#undef SPECIALIZE_SET_UNIFORM_VECTORS

#define SPECIALIZE_SET_UNIFORM_MATRICES(TYPE, GL_POSTFIX) \
template <> \
inline void program::set_uniform       (const std::string& name, const TYPE& value) \
{ \
  auto location = get_uniform_location(name); \
  if (location < 0) \
    return; \
  glUniformMatrix##GL_POSTFIX##v(location, 1, GL_FALSE, glm::value_ptr(value)); \
} \
template <> \
inline void program::set_uniform_array (const std::string& name, const std::vector<TYPE>& values) \
{ \
  auto location = get_uniform_location(name); \
  if (location < 0) \
    return; \
  glUniformMatrix##GL_POSTFIX##v(location, (GLsizei) values.size(), GL_FALSE, glm::value_ptr(values[0])); \
} \

SPECIALIZE_SET_UNIFORM_MATRICES(glm::fmat2, 2f)
SPECIALIZE_SET_UNIFORM_MATRICES(glm::fmat3, 3f)
SPECIALIZE_SET_UNIFORM_MATRICES(glm::fmat4, 4f)

#undef SPECIALIZE_SET_UNIFORM_MATRICES
}

#endif