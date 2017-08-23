#ifndef PLI_VIS_SPHERE_TESSELLATION_H_
#define PLI_VIS_SPHERE_TESSELLATION_H_

#include <vector>

#include <vector_types.h>

#include <pli_vis/cuda/sh/convert.h>
#include <pli_vis/cuda/sh/vector_ops.h>

namespace pli
{
template<typename vector_precision>
struct polyhedron
{
  std::vector<vector_precision> vertices;
  std::vector<vector_precision> indices ;
};

template<typename scalar_precision = float, typename vector_precision = float3>
polyhedron<vector_precision> make_icosahedron()
{
  const scalar_precision x(0.525731112119133606);
  const scalar_precision z(0.850650808352039932);
  return polyhedron<vector_precision>
  {
    {
      {-x, 0, z}, { x, 0,  z}, {-x,  0, -z}, { x,  0, -z},
      { 0, z, x}, { 0, z, -x}, { 0, -z,  x}, { 0, -z, -x},
      { z, x, 0}, {-z, x,  0}, { z, -x,  0}, {-z, -x,  0}
    },
    {
      {0,  4,  1}, {0, 9,  4}, {9,  5, 4}, { 4, 5, 8}, {4, 8,  1},
      {8, 10,  1}, {8, 3, 10}, {5,  3, 8}, { 5, 2, 3}, {2, 7,  3},
      {7, 10,  3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1,  6},
      {6,  1, 10}, {9, 0, 11}, {9, 11, 2}, { 9, 2, 5}, {7, 2, 11}
    }
  };
}

// Normals are in spherical coordinates.
template<typename vector_precision = float3>
void tessellate_triangle_normals(
  const vector_precision&        v1, 
  const vector_precision&        v2, 
  const vector_precision&        v3, 
  const std::size_t&             depth, 
  std::vector<vector_precision>& normals)
{
  if(depth == 0)
  {
    normals.push_back(to_spherical_coords(normalize(cross(v2 - v1, v3 - v1))));
    return;
  }

  auto v12 = normalize(v1 + v2);
  auto v23 = normalize(v2 + v3);
  auto v31 = normalize(v3 + v1);

  tessellate_triangle_normals(v1 , v12, v31, depth - 1, normals);
  tessellate_triangle_normals(v2 , v23, v12, depth - 1, normals);
  tessellate_triangle_normals(v3 , v31, v23, depth - 1, normals);
  tessellate_triangle_normals(v12, v23, v31, depth - 1, normals);
}

// Normals are in spherical coordinates.
template<typename vector_precision = float3>
std::vector<vector_precision> tessellate_polyhedron_normals(
  const polyhedron<vector_precision>& polyhedron, 
  const std::size_t&                  depth     )
{
  std::vector<vector_precision> normals;
  for(auto i = 0; i < polyhedron.indices.size(); i++)
    tessellate_triangle_normals(
      polyhedron.vertices[polyhedron.indices[i].x],
      polyhedron.vertices[polyhedron.indices[i].y],
      polyhedron.vertices[polyhedron.indices[i].z],
      depth  ,
      normals);
  return normals;
}
}

#endif
