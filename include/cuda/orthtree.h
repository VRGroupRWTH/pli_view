#ifndef ORTHTREE_H_
#define ORTHTREE_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <math.h>
#include <type_traits>

#ifdef ORTHTREE_CUDA_SUPPORT
#include <vector_types.h>
#endif

// Functions for creating and manipulating a complete (dense) linear orthtree.
namespace orthtree
{
  // Utility functions.
  inline std::size_t to_linear_index       (std::initializer_list<std::size_t> indices)
  {
    // Note that last dimension is indexed first.
    std::size_t index = 0;
    for (auto i = 0; i < indices.size(); ++i)
      index += *(indices.begin() + i) * powf(2, indices.size() - 1 - i);
    return index;
  }
  template<std::size_t dimensions, typename array_type>
  array_type         to_n_dimensional_index(std::size_t index)
  {
    array_type indices; 
    to_n_dimensional_index<dimensions>(index, indices);
    return indices;
  }
  template<std::size_t dimensions, typename array_type>
  void               to_n_dimensional_index(std::size_t index, array_type& indices)
  {
    // Note that last dimension is indexed first.
    for (auto i = 0; i < dimensions; ++i)
    {
      indices[i] = index      / powf(2, dimensions - 1 - i);
      index     -= indices[i] * powf(2, dimensions - 1 - i);
    }
  }

  // Tree functions.
  template<std::size_t dimensions>
  std::size_t subdivision_count   ()
  {
    return powf(2, dimensions);
  }
  template<std::size_t dimensions>
  std::size_t depth_size          (std::size_t depth)
  {
    return powf(2, dimensions * depth);
  }
  template<std::size_t dimensions> 
  std::size_t depth_start         (std::size_t depth)
  {
    return (powf(2, dimensions * depth) - 1.0) / (subdivision_count<dimensions>() - 1);
  }
  template<std::size_t dimensions>
  std::size_t depth_end           (std::size_t depth)
  {
    return depth_start<dimensions>(depth) + depth_size<dimensions>(depth);
  }
  template<std::size_t dimensions>
  std::size_t node_count          (std::size_t max_depth)
  {
    return depth_start<dimensions>(max_depth + 1);
  }

  // Node functions.
  template<std::size_t dimensions>
  std::size_t depth               (std::size_t index)
  {
    return floor(logf(index * (subdivision_count<dimensions>() - 1) + 1) / logf(subdivision_count<dimensions>()));
  }
  template<std::size_t dimensions, typename precision>
  precision   half_size           (std::size_t index)
  {
    return 0.5 / powf(2, depth<dimensions>(index));
  }
  template<std::size_t dimensions>
  std::size_t parent_index        (std::size_t index)
  {
    return floor(static_cast<float>(index - 1) / subdivision_count<dimensions>());
  }
  template<std::size_t dimensions>
  std::size_t child_index         (std::size_t index, std::size_t                        child_local_index  )
  {
    return subdivision_count<dimensions>() * index + child_local_index + 1;
  }
  template<std::size_t dimensions>
  std::size_t child_index         (std::size_t index, std::initializer_list<std::size_t> child_local_indices)
  {
    return child_index<dimensions>(index, to_linear_index(child_local_indices));
  }
  template<std::size_t dimensions>
  std::size_t index_in_parent     (std::size_t index)
  {
    auto parent = parent_index<dimensions>(index);
    for (auto i = 0; i < subdivision_count<dimensions>(); i++)
      if (child_index<dimensions>(parent, i) == index)
        return i;
    return 0;
  }
  template<std::size_t dimensions, typename array_type>
  array_type  index_in_parent     (std::size_t index)
  {
    return to_n_dimensional_index<dimensions, array_type>(index_in_parent<dimensions>(index));
  }
  template<std::size_t dimensions, typename index_type, typename array_type>
  array_type  position            (std::size_t index)
  {
    array_type array;
    position<dimensions, index_type>(index, array);
    return array;
  }
  template<std::size_t dimensions, typename index_type, typename array_type>
  void        position            (std::size_t index, array_type& array)
  {
    typedef typename std::remove_reference<decltype(array[0])>::type element_type;

    if (index == 0)
    {
      for (auto i = 0; i < dimensions; i++)
        array[i] = 0;
      return;
    }

    auto parent_position = position<dimensions, index_type, array_type>(parent_index<dimensions>(index));
    auto index_in_parent = orthtree::index_in_parent<dimensions, index_type  >(index);
    auto half_size       = orthtree::half_size      <dimensions, element_type>(index);
    for (auto i = 0; i < dimensions; i++)
      array[i] = parent_position[i] + (index_in_parent[i] != 0 ? half_size : -half_size);
  }

  // Note that last dimension is indexed first.
#ifdef ORTHTREE_CUDA_SUPPORT

  #define SPECIALIZE_ORTHTREE(TYPE)                                                                     \
  template<std::size_t dimensions>                                                                      \
  void to_n_dimensional_index(std::size_t index, TYPE& indices)                                         \
  {                                                                                                     \
    auto indices_ptr = &indices.x;                                                                      \
    for (auto i = 0; i < dimensions; ++i)                                                               \
    {                                                                                                   \
      indices_ptr[i] = index / powf(2, dimensions - 1 - i);                                             \
      index         -= indices_ptr[i] * powf(2, dimensions - 1 - i);                                    \
    }                                                                                                   \
  }                                                                                                     \
  template<std::size_t dimensions, typename index_type>                                                 \
  void position(std::size_t index, TYPE& array)                                                         \
  {                                                                                                     \
    typedef typename std::remove_reference<decltype(array.x)>::type element_type;                       \
                                                                                                        \
    auto array_ptr = &array.x;                                                                          \
                                                                                                        \
    if (index == 0)                                                                                     \
    {                                                                                                   \
      for (auto i = 0; i < dimensions; i++)                                                             \
        array_ptr[i] = 0;                                                                               \
      return;                                                                                           \
    }                                                                                                   \
                                                                                                        \
    auto parent_position     = position<dimensions, index_type, TYPE>(parent_index<dimensions>(index)); \
    auto parent_ptr          = &parent_position.x;                                                      \
    auto index_in_parent     = orthtree::index_in_parent<dimensions, index_type  >(index);              \
    auto index_in_parent_ptr = &index_in_parent.x;                                                      \
    auto half_size           = orthtree::half_size      <dimensions, element_type>(index);              \
    for (auto i = 0; i < dimensions; i++)                                                               \
      array_ptr[i] = parent_ptr[i] + (index_in_parent_ptr[i] != 0 ? half_size : -half_size);            \
  }                                                                                                     \

  SPECIALIZE_ORTHTREE(char2     )
  SPECIALIZE_ORTHTREE(uchar2    )
  SPECIALIZE_ORTHTREE(short2    )
  SPECIALIZE_ORTHTREE(ushort2   )
  SPECIALIZE_ORTHTREE(int2      )
  SPECIALIZE_ORTHTREE(uint2     )
  SPECIALIZE_ORTHTREE(long2     )
  SPECIALIZE_ORTHTREE(ulong2    )
  SPECIALIZE_ORTHTREE(float2    )
  SPECIALIZE_ORTHTREE(longlong2 )
  SPECIALIZE_ORTHTREE(ulonglong2)
  SPECIALIZE_ORTHTREE(double2   )
                                       
  SPECIALIZE_ORTHTREE(char3     )
  SPECIALIZE_ORTHTREE(uchar3    )
  SPECIALIZE_ORTHTREE(short3    )
  SPECIALIZE_ORTHTREE(ushort3   )
  SPECIALIZE_ORTHTREE(int3      )
  SPECIALIZE_ORTHTREE(uint3     )
  SPECIALIZE_ORTHTREE(long3     )
  SPECIALIZE_ORTHTREE(ulong3    )
  SPECIALIZE_ORTHTREE(float3    )
  SPECIALIZE_ORTHTREE(longlong3 )
  SPECIALIZE_ORTHTREE(ulonglong3)
  SPECIALIZE_ORTHTREE(double3   )

  SPECIALIZE_ORTHTREE(char4     )
  SPECIALIZE_ORTHTREE(uchar4    )
  SPECIALIZE_ORTHTREE(short4    )
  SPECIALIZE_ORTHTREE(ushort4   )
  SPECIALIZE_ORTHTREE(int4      )
  SPECIALIZE_ORTHTREE(uint4     )
  SPECIALIZE_ORTHTREE(long4     )
  SPECIALIZE_ORTHTREE(ulong4    )
  SPECIALIZE_ORTHTREE(float4    )
  SPECIALIZE_ORTHTREE(longlong4 )
  SPECIALIZE_ORTHTREE(ulonglong4)
  SPECIALIZE_ORTHTREE(double4   )

  #undef SPECIALIZE_ORTHTREE

#endif
}

#endif
