#ifndef PLI_VIS_FDM_POLY_DATA_HPP_
#define PLI_VIS_FDM_POLY_DATA_HPP_

#include <array>
#include <vector>

#include <boost/multi_array.hpp>
#include <vtkCellArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

#include <graphics/color_mappers/rgb.hpp>

namespace pli
{
class fdm_factory
{
public:
  template<typename points_type, typename indices_type = unsigned, typename color_mapper_type = rgb_color_mapper>
  static vtkSmartPointer<vtkPolyData> create(
    const std::vector<points_type>&            points       ,
    const std::vector<indices_type>&           indices      ,
    color_mapper_type                          color_mapper = color_mapper_type())
  {
    auto poly_data = vtkSmartPointer<vtkPolyData>         ::New();
    auto positions = vtkSmartPointer<vtkPoints>           ::New();
    auto colors    = vtkSmartPointer<vtkUnsignedCharArray>::New();
    auto cells     = vtkSmartPointer<vtkCellArray>        ::New();
    colors   ->SetNumberOfComponents(3);
    positions->SetNumberOfPoints    (points.size());
    colors   ->SetNumberOfTuples    (points.size());
    cells    ->Allocate             (cells->EstimateSize(points.size(), 4));

    for (auto i = 0; i < points.size(); i++)
    {
      positions->SetPoint(i, points[i].data());
      colors   ->SetTuple(i, (color_mapper.template map<points_type>(points[i])).data());
    }

    for (auto i = 0; i < indices.size(); i+=4)
    {
      vtkIdType vtk_indices[4] = { indices[i], indices[i + 1], indices[i + 2], indices[i + 3] };
      cells->InsertNextCell(4, vtk_indices);
    }

    poly_data->SetPoints(positions);
    poly_data->SetPolys (cells);
    poly_data->GetPointData()->SetScalars(colors);
    return poly_data;
  }
  
  template<typename points_type, typename indices_type = unsigned, typename color_mapper_type = rgb_color_mapper>
  static vtkSmartPointer<vtkPolyData> create(
          boost::multi_array<points_type , 4>& points        ,
    const boost::multi_array<indices_type, 4>& indices       ,
    const points_type&                         vector_spacing,
    const std::array<std::size_t, 3>           block_size    ,
    color_mapper_type                          color_mapper  = color_mapper_type())
  {
    auto poly_data      = vtkSmartPointer<vtkPolyData>         ::New();
    auto positions      = vtkSmartPointer<vtkPoints>           ::New();
    auto colors         = vtkSmartPointer<vtkUnsignedCharArray>::New();
    auto cells          = vtkSmartPointer<vtkCellArray>        ::New();
    auto num_elements   = points.num_elements();
    auto num_components = 3;
    colors   ->SetNumberOfComponents(num_components);
    positions->SetNumberOfPoints    (num_elements  );
    colors   ->SetNumberOfTuples    (num_elements  );
    cells    ->Allocate             (cells->EstimateSize(num_elements, 4));

    std::array<float, 3> offset  = {
      vector_spacing[0] * (block_size[0] - 1) * 0.5,
      vector_spacing[1] * (block_size[1] - 1) * 0.5,
      vector_spacing[2] * (block_size[2] - 1) * 0.5};
    std::array<float, 3> spacing = {
      vector_spacing[0] * block_size[0],
      vector_spacing[1] * block_size[1],
      vector_spacing[2] * block_size[2] };
    auto scale = spacing[0] * 0.5;

    auto shape      = points.shape();
    auto points_ptr = points.data ();
    for (auto i = 0; i < points.num_elements(); i++)
    {
      auto position = points_ptr[i];
      colors->SetTuple(i, (color_mapper.template map<points_type>(position)).data());

      // Scale by primary axis.
      position[0] *= scale;
      position[1] *= scale;
      position[2] *= scale;

      // Position by offset, spacing and index.
      auto index = int(i / shape[3]);
      position[0] += offset[0] + spacing[0] * (index / (shape[1] * shape[2]));
      position[1] += offset[1] + spacing[1] * ((index / shape[2]) % shape[1]);
      position[2] += offset[2] + spacing[2] * ((index % shape[2]));

      positions->SetPoint(i, position.data());
    }

    auto indices_ptr = indices.data();
    for (auto i = 0; i < indices.num_elements(); i += 4)
    {
      vtkIdType vtk_indices[4] = {
        indices_ptr[i],
        indices_ptr[i + 1],
        indices_ptr[i + 2],
        indices_ptr[i + 3] };
      cells->InsertNextCell(4, vtk_indices);
    }

    poly_data->SetPoints(positions);
    poly_data->SetPolys (cells);
    poly_data->GetPointData()->SetScalars(colors);
    return poly_data;
  }
};
}

#endif