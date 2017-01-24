#ifndef PLI_VIS_FOM_POLY_DATA_HPP_
#define PLI_VIS_FOM_POLY_DATA_HPP_

#define _USE_MATH_DEFINES 

#include <array>
#include <math.h>

#include <boost/multi_array.hpp>

#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include <convert.hpp>

#include <utility/vtk/color_mappers/rgb.hpp>

namespace pli
{
class fom_factory
{
public:
  template<typename scalar_type, typename color_mapper_type = rgb_color_mapper>
  static vtkSmartPointer<vtkPolyData> create(
    const boost::multi_array<scalar_type, 3>& directions    , 
    const boost::multi_array<scalar_type, 3>& inclinations  ,
    const std::array<scalar_type, 3>&         voxel_size    ,
    bool                                      input_radians = false,
    color_mapper_type                         color_mapper  = color_mapper_type())
  {
    auto poly_data      = vtkSmartPointer<vtkPolyData>          ::New();
    auto positions      = vtkSmartPointer<vtkPoints>            ::New();
    auto rotations      = vtkSmartPointer<vtkDoubleArray>       ::New();
    auto colors         = vtkSmartPointer<vtkUnsignedCharArray> ::New();
    auto num_components = 3;
    auto num_elements   = directions.num_elements();
    rotations->SetNumberOfComponents(num_components);
    colors   ->SetNumberOfComponents(num_components);
    positions->SetNumberOfPoints    (num_elements  );
    rotations->SetNumberOfTuples    (num_elements  );
    colors   ->SetNumberOfTuples    (num_elements  );

    auto index = 0;
    auto shape = directions.shape();
    for (auto x = 0; x < shape[0]; x++) 
    {
      for (auto y = 0; y < shape[1]; y++) 
      {
        for (auto z = 0; z < shape[2]; z++) 
        {
          std::array<float, 3> position = {voxel_size[0] * x, voxel_size[1] * y, voxel_size[2] * z};
          std::array<float, 3> rotation = {0.5};

          if (input_radians)
          {
            rotation[1] = (M_PI / 2 + directions  [x][y][z]);
            rotation[2] = (M_PI / 2 - inclinations[x][y][z]);
          }
          else
          {
            rotation[1] = (90.0 + directions  [x][y][z]) * M_PI / 180.0;
            rotation[2] = (90.0 - inclinations[x][y][z]) * M_PI / 180.0;
          }

          rotation = to_cartesian_coords(rotation);
          positions->SetPoint(index, position.data());
          rotations->SetTuple(index, rotation.data());
          colors   ->SetTuple(index, color_mapper.template map<scalar_type>(rotation).data());

          index++;
        }
      }
    }

    poly_data->SetPoints(positions);
    poly_data->GetPointData()->SetVectors(rotations);
    poly_data->GetPointData()->SetScalars(colors);
    return poly_data;
  }
};
}

#endif