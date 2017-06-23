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
#ifndef H5CONVERTER_MISC_HPP
#define H5CONVERTER_MISC_HPP

#include <string>
#include <sstream>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cassert>

#ifdef H5_USE_BOOST_MULTI_ARRAY
#include <boost/multi_array.hpp>
#endif

#ifdef H5_USE_BOOST_UBLAS
#include <boost/numeric/ublas/matrix.hpp>
#endif

#include <H5Dpublic.h>
#include <H5Ppublic.h>

#include "../H5DataType.hpp"
#include "../H5DataSpace.hpp"

#include "H5Utils.hpp"

namespace HighFive{
  
namespace details{

inline void check_dimensions_vector(size_t size_vec, size_t size_dataset, size_t dimension){
    if( size_vec != size_dataset){
        std::ostringstream ss;
        ss << "Mismatch between vector size ("<< size_vec << ") and dataset size (" << size_dataset;
        ss << ") on dimension " << dimension;
        throw DataSetException(ss.str());
    }
}

// copy multi dimensional vector in C++ in one C-style multi dimensional buffer
template<typename T>
inline void vectors_to_single_buffer(const std::vector<T> & vec_single_dim,
                              const std::vector<size_t> & dims,
                              size_t current_dim,
                              std::vector<T> & buffer){

    check_dimensions_vector(vec_single_dim.size(), dims[current_dim], current_dim);
    buffer.insert(buffer.end(), vec_single_dim.begin(), vec_single_dim.end());
}

template<typename T>
inline void vectors_to_single_buffer(const std::vector<T> & vec_multi_dim,
                              const std::vector<size_t> & dims,
                              size_t current_dim,
                              std::vector< typename type_of_array<T>::type > & buffer){

    check_dimensions_vector(vec_multi_dim.size(), dims[current_dim], current_dim);
    for(typename std::vector<T>::const_iterator it = vec_multi_dim.begin(); it < vec_multi_dim.end(); ++it){
        vectors_to_single_buffer(*it, dims, current_dim+1, buffer);
    }
}


// copy single buffer to multi dimensional vector, following dimensions specified
template<typename T>
inline typename std::vector<T>::iterator single_buffer_to_vectors(
                              typename std::vector<T>::iterator begin_buffer,
                              typename std::vector<T>::iterator end_buffer,
                              const std::vector<size_t> & dims,
                              size_t current_dim,
                              std::vector<T> & vec_single_dim){
    const size_t n_elems = dims[current_dim];
    typename std::vector<T>::iterator end_copy_iter = std::min(begin_buffer+ n_elems, end_buffer);
    vec_single_dim.assign(begin_buffer, end_copy_iter);
    return end_copy_iter;
}

template<typename T, typename U>
inline typename std::vector<T>::iterator single_buffer_to_vectors(
                             typename std::vector<T>::iterator  begin_buffer,
                             typename std::vector<T>::iterator  end_buffer,
                             const std::vector<size_t> & dims,
                             size_t current_dim,
                             std::vector<U> & vec_multi_dim){

    const size_t n_elems = dims[current_dim];
    vec_multi_dim.resize(n_elems);

    for(typename std::vector<U>::iterator it = vec_multi_dim.begin(); it < vec_multi_dim.end(); ++it){
        begin_buffer = single_buffer_to_vectors(begin_buffer, end_buffer, dims, current_dim+1, *it);
    }
    return begin_buffer;
}


// apply conversion operations to the incoming data
template<typename Array, class Enable = void>
struct data_converter{
    inline data_converter(Array & datamem, DataSpace & space){
        (void) datamem; (void) space; // do nothing
    }

    inline Array & transform_read (Array & datamem) { return datamem; }

    inline Array & transform_write (Array & datamem) { return datamem; }

    inline void process_result(Array & datamem){
        (void) datamem; // do nothing
    }
};

// apply conversion for vectors 1D
template<typename T >
struct data_converter<std::vector<T>, typename enable_if< (is_same<T, typename type_of_array<T>::type>::value) >::type > {
    inline data_converter(std::vector< T > & vec, DataSpace & space, size_t dim = 0) : _space(&space), _dim(dim){
        assert(_space->getDimensions().size() > dim);
        (void) vec;
    }

    inline typename type_of_array<T>::type*  transform_read(std::vector<T> & vec) {
            vec.resize(_space->getDimensions()[_dim]);
            return &(vec[0]);
    }

    inline typename type_of_array<T>::type*  transform_write(std::vector<T> & vec) { return &(vec[0]); }

    inline void process_result(std::vector<T> & vec){
         (void) vec;
    }

    DataSpace* _space;
    size_t _dim;
};

// apply conversion for arrays 1D
template<typename T, std::size_t Size>
struct data_converter<std::array<T, Size>, typename enable_if< (is_same<T, typename type_of_array<T>::type>::value) >::type >{
  inline data_converter(std::array<T, Size> & array, DataSpace & space, size_t dim = 0) : _space(&space), _dim(dim){
    assert(_space->getDimensions().size() > dim);
    (void) array;
  }

  inline typename type_of_array<T>::type*  transform_read(std::array<T, Size> & array) {
    assert(array.size() == _space->getDimensions()[_dim]);
    return &(array[0]);
  }

  inline typename type_of_array<T>::type*  transform_write(std::array<T, Size> & array) { return &(array[0]); }

  inline void process_result(std::array<T, Size> & array){
    (void)array;
  }

  DataSpace* _space;
  size_t _dim;
};

#ifdef H5_USE_BOOST_MULTI_ARRAY
// apply conversion to boost multi array
template<typename T, std::size_t Dims>
struct data_converter< boost::multi_array<T, Dims>, void >{

    typedef typename boost::multi_array<T, Dims> MultiArray;

    inline data_converter(MultiArray & array, DataSpace & space, size_t dim = 0) : _dims(space.getDimensions()){
        assert(_dims.size() == Dims);
        (void) dim;
        (void) array;
    }

    inline /* typename type_of_array<T>::type* */ T* transform_read(MultiArray & array) {
            if(std::equal(_dims.begin(), _dims.end(), array.shape()) == false){
                boost::array<typename MultiArray::index, Dims> ext;
                std::copy(_dims.begin(), _dims.end(), ext.begin());
                array.resize(ext);
            }
            return array.data();
    }

    inline /* typename type_of_array<T>::type* */ T* transform_write(MultiArray & array) { return array.data(); }

    inline void process_result(MultiArray & array){
         (void) array;
    }

    std::vector<size_t> _dims;
};

// apply conversion to boost multi array view
template<typename T, std::size_t Dims>
struct data_converter< boost::detail::multi_array::multi_array_view<T, Dims>, void >{

  typedef boost::detail::multi_array::multi_array_view<T, Dims> MultiArrayView;

  inline data_converter(MultiArrayView & array, DataSpace & space, size_t dim = 0) : _dims(space.getDimensions()){
    assert(_dims.size() == Dims);
    (void)dim;
    (void)array;
  }

  inline typename type_of_array<T>::type* transform_read (MultiArrayView & array) { return array.origin(); }

  inline typename type_of_array<T>::type* transform_write(MultiArrayView & array) { return array.origin(); }

  inline void process_result(MultiArrayView & array){
    (void)array;
  }

  std::vector<size_t> _dims;
};

// apply conversion to boost sub array
template<typename T, std::size_t Dims>
struct data_converter< boost::detail::multi_array::sub_array<T, Dims>, void >{

  typedef boost::detail::multi_array::sub_array<T, Dims> SubArray;

  inline data_converter(SubArray & array, DataSpace & space, size_t dim = 0) : _dims(space.getDimensions()){
    assert(_dims.size() == Dims);
    (void)dim;
    (void)array;
  }

  inline typename type_of_array<T>::type*  transform_read (SubArray & array) { return array.origin(); }

  inline typename type_of_array<T>::type*  transform_write(SubArray & array) { return array.origin(); }

  inline void process_result(SubArray & array){
    (void)array;
  }

  std::vector<size_t> _dims;
};
#endif

#ifdef H5_USE_BOOST_UBLAS
// apply conversion to boost matrix ublas
template<typename T>
struct data_converter< boost::numeric::ublas::matrix<T> , void >{

    typedef typename boost::numeric::ublas::matrix<T> Matrix;

    inline data_converter(Matrix & array, DataSpace & space, size_t dim = 0) : _dims(space.getDimensions()){
        assert(_dims.size() == 2);
        (void) dim;
        (void) array;
    }

    inline typename type_of_array<T>::type*  transform_read(Matrix & array) {
            boost::array<std::size_t, 2> sizes = {{ array.size1(), array.size2() }};

            if(std::equal(_dims.begin(), _dims.end(), sizes.begin()) == false){
                array.resize(_dims[0], _dims[1], false);
                array(0,0 ) = 0; // force initialization
            }

            return &(array(0,0));
    }

    inline typename type_of_array<T>::type*  transform_write(Matrix & array) { return &(array(0,0)); }

    inline void process_result(Matrix & array){
         (void) array;
    }

    std::vector<size_t> _dims;
};
#endif



// apply conversion for vectors nested vectors
template<typename T >
struct data_converter<std::vector<T>, typename enable_if< (is_container<T>::value) >::type > {
    inline data_converter(std::vector< T > & vec, DataSpace & space, size_t dim = 0) : _dims(space.getDimensions()), _dim(dim),
        _vec_align()
    {
        (void) vec;
    }

    inline typename type_of_array<T>::type* transform_read(std::vector<T> & vec) {
        (void) vec;
        _vec_align.resize(get_total_size());
        return &(_vec_align[0]);
    }

    inline typename type_of_array<T>::type*  transform_write(std::vector<T> & vec) {
        _vec_align.reserve(get_total_size());
        vectors_to_single_buffer<T>(vec, _dims, 0, _vec_align);
        return &(_vec_align[0]);

    }

    inline void process_result(std::vector<T> & vec){
         single_buffer_to_vectors<typename type_of_array<T>::type, T>(_vec_align.begin(), _vec_align.end(), _dims,
                                  0, vec);
    }

    inline size_t get_total_size(){
        return std::accumulate(_dims.begin(), _dims.end(), 1, std::multiplies<size_t>());
    }


    std::vector<size_t> _dims;
    size_t _dim;
    std::vector< typename type_of_array<T>::type > _vec_align;
};

// apply conversion for vectors of string (derefence)
template<>
struct data_converter<std::vector<std::string>, void>{
    inline data_converter(std::vector<std::string> & vec, DataSpace & space) : _space(space) {
        (void) vec;
    }

    // create a C vector adapted to HDF5
    // fill last element with NULL to identify end
    inline char**  transform_read(std::vector<std::string> & vec) {
        (void) vec;
        _c_vec.resize(_space.getDimensions()[0], NULL);
        return (&_c_vec[0]);
    }

    static inline char* char_converter(const std::string & str){
        return const_cast<char*>(str.c_str());
    }

    inline char**  transform_write(std::vector<std::string> & vec) {
        _c_vec.resize(vec.size()+1, NULL);
        std::transform(vec.begin(), vec.end(), _c_vec.begin(), &char_converter);
        return (&_c_vec[0]);
    }

    inline void process_result(std::vector<std::string> & vec){
        (void) vec;
        vec.resize(_c_vec.size());
        for(size_t i = 0; i < vec.size(); ++i){
            vec[i] = std::string(_c_vec[i]);
        }

        if(_c_vec.empty() == false && _c_vec[0] != NULL){
             AtomicType<std::string > str_type;
            (void) H5Dvlen_reclaim(str_type.getId(), _space.getId(), H5P_DEFAULT, &(_c_vec[0]));
        }

    }

    std::vector<char*> _c_vec;
    DataSpace &  _space;

};

}
}

#endif // H5CONVERTER_MISC_HPP
