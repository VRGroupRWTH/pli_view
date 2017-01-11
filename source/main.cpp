#include <chrono>
#include <iostream>

#include <hdf5/hdf5_io.hpp>

/*
void main()
{
  std::cout << "Starting PLI visualization app." << std::endl;

  pli::hdf5_io<float> io("D:/data/Vervet1818/Vervet1818.h5");
  std::string dataset_path_prefix = "%Slice%/Microscope/Processed/Registered/";
  io.set_attribute_path_voxel_size     ("DataSpacing");
  io.set_dataset_path_mask             (dataset_path_prefix + "Mask");
  io.set_dataset_path_transmittance    (dataset_path_prefix + "NTransmittance");
  io.set_dataset_path_retardation      (dataset_path_prefix + "Retardation");
  io.set_dataset_path_fiber_direction  (dataset_path_prefix + "Direction");
  io.set_dataset_path_fiber_inclination(dataset_path_prefix + "Inclination");

  std::array<float, 3> voxel_size;
  io.load_voxel_size(voxel_size);

  std::array<std::size_t, 3> offset = {{45000, 35000, 536}};
  std::array<std::size_t, 3> size   = {{  500,   500,   3}};

  boost::multi_array<float, 3> fiber_direction_map;
  io.load_fiber_direction_map(offset, size, fiber_direction_map);

  boost::multi_array<float, 3> fiber_inclination_map;
  io.load_fiber_inclination_map(offset, size, fiber_inclination_map);

  std::cout << "Ending PLI visualization app." << std::endl;
}
*/

void main()
{
  std::cout << "Starting PLI visualization app." << std::endl;

  pli::hdf5_io<float> io("D:/data/Test/Test.h5");
  io.set_dataset_path_fiber_direction   ("fiber_directions");
  io.set_dataset_path_fiber_distribution("fiber_distributions");

  boost::multi_array<float, 3> fiber_direction_map;
  io.load_fiber_direction_map({{0, 0, 0}}, {{5, 5, 1}}, fiber_direction_map);
  std::for_each(fiber_direction_map.data(), fiber_direction_map.data() + fiber_direction_map.num_elements(), [](float& elem) { elem++; });
  io.save_fiber_direction_map({{0, 0, 0}}, {{5, 5, 1}}, fiber_direction_map);

  std::cout << "Ending PLI visualization app." << std::endl;
}