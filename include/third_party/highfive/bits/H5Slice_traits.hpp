/*
 * Copyright (C) 2015 Adrien Devresse <adrien.devresse@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */
#ifndef H5SLICE_TRAITS_HPP
#define H5SLICE_TRAITS_HPP

#include <boost/optional.hpp>

namespace HighFive{

class DataSet;
class Group;
class DataSpace;
class DataType;
class Selection;

template<typename Derivate>
class SliceTraits{
public:


    ///
    /// select a region in the current Slice/Dataset of 'count' points at 'offset'
    /// vector offset and count have to be from the same dimension
    ///
    Selection select(const std::vector<size_t> & offset, const std::vector<size_t> & count, boost::optional<std::vector<size_t>> stride = boost::none) const;

    ///
    /// select a region in the current Slice/Dataset of 'count' points at 'offset'
    /// array offset and count have to be from the same dimension
    ///
    template<std::size_t Size>
    Selection select(const std::array<size_t, Size> & offset, const std::array<size_t, Size> & count, boost::optional<std::array<size_t, Size>> stride = boost::none) const;

    ///
    /// Read the entire dataset into a buffer
    /// An exception is raised is if the numbers of dimension of the buffer and of the dataset are different
    ///
    /// The array type can be a N-pointer or a N-vector ( e.g int** integer two dimensional array )
    template <typename T>
    void read(T & array) const;

    ///
    /// Write the integrality N-dimension buffer to this dataset
    /// An exception is raised is if the numbers of dimension of the buffer and of the dataset are different
    ///
    /// The array type can be a N-pointer or a N-vector ( e.g int** integer two dimensional array )
    template <typename T>
    void write(T & buffer);


private:
    typedef Derivate derivate_type;
};

}


#endif // H5SLICE_TRAITS_HPP
