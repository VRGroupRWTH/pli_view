#ifndef PLI_VIS_VECTOR_FIELD_HPP_
#define PLI_VIS_VECTOR_FIELD_HPP_

#define _USE_MATH_DEFINES

#include <array>
#include <cassert>
#include <math.h>
#include <memory>

#include <boost/multi_array.hpp>

#include <all.hpp>
#include <decorators.h>

#include <vector_types.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

namespace pli
{
// Receives two 3D scalar volumes corresponding to directions and inclinations respectively.
// Transfers them to the GPU.
// Creates a symmetric unit vector field using Cuda.
// Renders them on demand.
class vector_field
{
public:
  void start  ()
  {
    vertex_array_ .reset(new gl::vertex_array);
    vertex_buffer_.reset(new gl::array_buffer);
  }
  void render ()
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    vertex_array_->bind  ();
    glDrawArrays(GL_LINES, 0, draw_count_);
    vertex_array_->unbind();
  }
  void destroy()
  {

  }

  void set_data(
    const boost::multi_array<float, 3>& directions  ,
    const boost::multi_array<float, 3>& inclinations,
    const std::array<float, 3>&         spacing     )
  {
    if (vertex_buffer_cuda != nullptr)
    {
      cudaGraphicsUnregisterResource(vertex_buffer_cuda);
      vertex_buffer_cuda = nullptr;
    }

    draw_count_ = 2 * sizeof(float3) * directions.num_elements();
    vertex_array_ ->bind    ();
    vertex_buffer_->bind    ();
    vertex_buffer_->allocate(draw_count_);
    vertex_buffer_->unbind  ();
    vertex_array_ ->unbind  ();
    cudaGraphicsGLRegisterBuffer(&vertex_buffer_cuda, vertex_buffer_->id(), cudaGraphicsMapFlagsWriteDiscard);

    float3* vertices_ptr;
    size_t  num_bytes   ;
    cudaGraphicsMapResources(1, &vertex_buffer_cuda, nullptr);
    cudaGraphicsResourceGetMappedPointer((void**)&vertices_ptr, &num_bytes, vertex_buffer_cuda);
    assert(draw_count_);

    // TODO: Create a symmetric unit vector field using Cuda.
    std::cout << "Allocating and copying vectors." << std::endl;
    thrust::device_vector<float> directions_vector  (directions  .num_elements());
    thrust::device_vector<float> inclinations_vector(inclinations.num_elements());
    copy_n(directions  .data(), directions  .num_elements(), directions_vector  .begin());
    copy_n(inclinations.data(), inclinations.num_elements(), inclinations_vector.begin());
    auto directions_ptr   = raw_pointer_cast(&directions_vector  [0]);
    auto inclinations_ptr = raw_pointer_cast(&inclinations_vector[0]);
    transform(directions_vector  .begin(), directions_vector  .end(), directions_vector  .begin(),       thrust::placeholders::_1  * M_PI / 180.0);
    transform(inclinations_vector.begin(), inclinations_vector.end(), inclinations_vector.begin(), (90 - thrust::placeholders::_1) * M_PI / 180.0);
    
    thrust::transform(directions_vector.begin(), directions_vector.end(), inclinations_vector.begin(), vertices_ptr, 
    [&] COMMON (const float& direction, const float& inclination) -> float3
    {
            
    });

    cudaGraphicsUnmapResources(1, &vertex_buffer_cuda, nullptr);
  }

private:
  std::unique_ptr<gl::vertex_array> vertex_array_      ;
  std::unique_ptr<gl::array_buffer> vertex_buffer_     ;
  cudaGraphicsResource*             vertex_buffer_cuda = nullptr;
  std::size_t                       draw_count_        = 0;
};
}

#endif