#ifndef PLI_VIS_RANDOM_TEXTURE_HPP_
#define PLI_VIS_RANDOM_TEXTURE_HPP_

#include <glm/glm.hpp>

#include <pli_vis/opengl/opengl.hpp>
#include <pli_vis/opengl/texture.hpp>

namespace pli
{
// Random texture is a 3D texture where each voxel is a random 4D vector.
class random_texture : public gl::texture_3d
{
public:
  random_texture(const glm::uvec3& size = glm::uvec3(1));
  random_texture(const random_texture&  that) = default;
  random_texture(      random_texture&& temp) = default;
  virtual ~random_texture()                   = default;

  random_texture& operator=(const random_texture&  that) = default;
  random_texture& operator=(      random_texture&& temp) = default;

  void generate(const glm::uvec3& size);
};
}

#endif
